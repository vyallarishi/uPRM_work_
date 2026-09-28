"""Stage 32 — uPRM TOY 1: can the base model localize the FIRST broken step?

WHY: every judge we built (stages 8/18/19/20/21, AUC 0.50-0.67 on contested)
asked "is this whole chain right?". uPRM (arXiv 2605.10158) asks a different
question -- WHERE does it first break -- parametrized as a single ordinal j
(j = T+1 means "no error anywhere"). That parametrization is self-consistent
by construction and is the one axis we have never tested.

THE MEASUREMENT (uPRM Eq. 6, INDEPENDENT scoring only -- joint scoring is
Toy 2 and must be run against a shuffled control first, per the stage19
peer-judge autopsy where a 0.92 headline was beaten 0.95 by a zero-model
answer-matching baseline):

    S(j) = 1[j <= T] * log p-_j  +  sum_{t<j} log p+_t

built from a grading transcript where each step is followed by a + or -
marker, reading renormalized P(+) vs P(-) at each marker slot.

TWO READOUTS, both needed:
  A) CLEAN-vs-BROKEN (the headline, comparable to every other judge here):
     score = S(j = T+1) - logsumexp_j S(j), i.e. the model's probability
     that the chain has NO error. AUC over gold-matching vs non-gold chains.
     Bar: 0.70 (the project's standing gate). Reference: whole-chain verdict
     judge 0.636 contested / 0.732 overall.
  B) LOCALIZATION (does argmax_j land on a real error?): on the subset of
     wrong chains containing a VERIFIABLE arithmetic slip -- a <<a op b=c>>
     whose arithmetic is false -- check whether argmax_j falls on the step
     containing the first such slip. Chance = 1/T. This is a real
     localization measurement, not a proxy.

CAVEAT (printed): Qwen3-14B-Base is a BASE model; uPRM used an
instruction-tuned grader and the +/- transcript is a chat format. A null
result here is partly a statement about base models, which is itself the
thing ICM's setup forces on us. No instruct sibling is on scratch.

Usage: --self-test | --dry-run | full via stage32_uprm.sbatch
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from decimal import Decimal, DivisionByZero, InvalidOperation
from pathlib import Path

from icm.experiments.q08_earlier_8b_study.rung3_reasoning_structure import _EQUATION_RE, _OP_SPLIT_RE, normalize_number
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, load_pools, match

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
# All 15 phaseA languages. English alone yields only 8 chains with a
# verifiable arithmetic slip (measured): at 14B, wrong English answers are
# almost always CONCEPTUAL errors with correct arithmetic throughout. Across
# 15 languages there are 277, which is what readout B needs.
LANGS = ("en", "de", "es", "fr", "ru", "hi", "ar", "ko", "fa",
         "am", "sw", "kn", "eu", "ml", "is")
MAX_STEPS = 12
MIN_STEPS = 2
CHAIN_MAX_CHARS = 1400
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 16
N_PER_CLASS = 350          # balanced clean/broken for readout A
N_SLIP_MIN = 250           # verifiable-slip chains to force into the sample
SEED = 20260824

SYSTEM = ("You are a strict mathematical reasoning judge. After each step you "
          "respond with + if the step is correct and - if it is incorrect.\n\n")


def split_steps(chain_text):
    """Reasoning steps = non-empty lines, excluding the '#### N' answer line."""
    steps = [line.strip() for line in chain_text.split("\n")
             if line.strip() and not line.lstrip().startswith("####")]
    return steps[:MAX_STEPS]


def eval_equation(lhs, rhs):
    """Is <<lhs=rhs>> arithmetically true? None if not evaluable."""
    operands = [normalize_number(tok) for tok in _OP_SPLIT_RE.split(lhs)]
    if not operands or any(o is None for o in operands):
        return None
    operators = _OP_SPLIT_RE.findall(lhs)
    if len(operators) != len(operands) - 1:
        return None
    try:
        total = Decimal(operands[0])
        for op, operand in zip(operators, operands[1:]):
            value = Decimal(operand)
            if op == "+":
                total += value
            elif op == "-":
                total -= value
            elif op in ("*", "×"):
                total *= value
            elif op in ("/", "÷"):
                if value == 0:
                    return None
                total /= value
            else:
                return None
        want = Decimal(rhs)
    except (InvalidOperation, DivisionByZero, ValueError):
        return None
    if total == want:
        return True
    scale = max(abs(total), abs(want))
    if scale == 0:
        return False
    return abs(total - want) / scale <= Decimal("0.005")


def first_bad_step(steps):
    """Index (1-based) of the first step containing false arithmetic, else None."""
    for index, step in enumerate(steps, start=1):
        for body in _EQUATION_RE.findall(step):
            if "=" not in body:
                continue
            lhs, _, rhs = body.rpartition("=")
            lhs = lhs.replace("×", "*").replace("÷", "/").replace(" ", "")
            right = normalize_number(rhs)
            if not lhs or right is None:
                continue
            verdict = eval_equation(lhs, right)
            if verdict is False:
                return index
    return None


def build_transcript(question, steps, j, pos=" +", neg=" -"):
    """Grading transcript asserting 'first error at step j' (j = len+1 means
    no error). Ends right after the final marker so every marker slot is a
    next-token position we can read."""
    lines = [SYSTEM + "Problem: " + question.strip()]
    for index, step in enumerate(steps, start=1):
        marker = neg if index == j else pos
        lines.append(f"Step {index}: {step}\nJudge:{marker}")
        if index == j:
            break                      # nothing after the first error matters
    return "\n\n".join(lines)


def marker_slots(question, steps, j):
    """Prompts whose NEXT token is the marker for each graded step, plus the
    marker we are asserting there. One forward pass per slot."""
    slots = []
    prefix = SYSTEM + "Problem: " + question.strip()
    for index, step in enumerate(steps, start=1):
        prompt = prefix + f"\n\nStep {index}: {step}\nJudge:"
        slots.append((prompt, index == j))
        marker = " -" if index == j else " +"
        prefix = prefix + f"\n\nStep {index}: {step}\nJudge:{marker}"
        if index == j:
            break
    return slots


def score_from_logprobs(step_logprobs, j, n_steps):
    """uPRM Eq. 6 from cached per-step (logp_plus, logp_minus).
    step_logprobs[t] is for step t+1 under the all-correct prefix."""
    total = 0.0
    for t in range(min(j - 1, n_steps)):
        total += step_logprobs[t][0]
    if j <= n_steps:
        total += step_logprobs[j - 1][1]
    return total


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- split_steps ---")
    check("drops the answer line",
          split_steps("a = 1\nb = 2\n#### 2"), ["a = 1", "b = 2"])
    check("drops blank lines", split_steps("a\n\n\nb"), ["a", "b"])

    print("--- eval_equation ---")
    check("true division", eval_equation("48/2", "24"), True)
    check("false division", eval_equation("48/2", "25"), False)
    check("true chain", eval_equation("6*2*3", "36"), True)
    check("false multiply", eval_equation("6*2", "14"), False)
    check("divide by zero -> None", eval_equation("5/0", "1"), None)
    check("tolerance holds", eval_equation("100/3", "33.333"), True)

    print("--- first_bad_step ---")
    good = ["Marcy has 6 * 2 = <<6*2=12>>12 tubes.",
            "That serves 12 * 3 = <<12*3=36>>36 people."]
    check("clean chain -> None", first_bad_step(good), None)
    bad = ["Marcy has 6 * 2 = <<6*2=12>>12 tubes.",
           "That serves 12 * 3 = <<12*3=30>>30 people.",
           "So 30 people."]
    check("finds the broken step", first_bad_step(bad), 2)
    check("no annotations -> None",
          first_bad_step(["He had 5 apples", "He ate 2"]), None)

    print("--- transcript / slots ---")
    steps = ["one", "two", "three"]
    t = build_transcript("Q?", steps, 2)
    check("marks step 1 correct", "Step 1: one\nJudge: +" in t, True)
    check("marks step 2 incorrect", "Step 2: two\nJudge: -" in t, True)
    check("stops after the first error", "Step 3" not in t, True)
    clean = build_transcript("Q?", steps, 4)
    check("j = T+1 marks everything correct", clean.count("Judge: +"), 3)
    check("j = T+1 has no minus", "Judge: -" in clean, False)
    slots = marker_slots("Q?", steps, 2)
    check("slots stop at the error", len(slots), 2)
    check("first slot asserts plus", slots[0][1], False)
    check("second slot asserts minus", slots[1][1], True)
    check("slot prompt ends at the marker",
          slots[0][0].endswith("Step 1: one\nJudge:"), True)

    print("--- Eq. 6 arithmetic ---")
    lp = [(-0.1, -2.0), (-0.2, -1.5), (-0.3, -1.0)]
    check("j=1 is just the minus at step 1",
          round(score_from_logprobs(lp, 1, 3), 4), -2.0)
    check("j=2 is plus1 + minus2",
          round(score_from_logprobs(lp, 2, 3), 4), round(-0.1 - 1.5, 4))
    check("j=T+1 is all pluses",
          round(score_from_logprobs(lp, 4, 3), 4), round(-0.6, 4))

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def build_sample(pools, rng):
    """Balanced sample for readout A (clean vs broken), with every available
    verifiable-slip chain force-included for readout B. Slip chains are a
    subset of `broken`, so A stays balanced by construction."""
    clean, broken = [], []
    for iid in sorted(pools):
        pool = pools[iid]
        if pool["lang"] not in LANGS:
            continue
        for chain in pool["chains"]:
            if len(chain["text"]) > CHAIN_MAX_CHARS:
                continue
            steps = split_steps(chain["text"])
            if len(steps) < MIN_STEPS:
                continue
            is_gold = match(chain["answer"], pool["gold"])
            row = {"iid": iid, "question": pool["question"], "steps": steps,
                   "is_gold": is_gold, "answer": chain["answer"],
                   "bad_step": first_bad_step(steps)}
            (clean if is_gold else broken).append(row)
    rng.shuffle(clean)
    rng.shuffle(broken)
    slips = [r for r in broken if r["bad_step"]][:max(N_SLIP_MIN, 0)]
    slip_ids = {id(r) for r in slips}
    rest = [r for r in broken if id(r) not in slip_ids]
    broken_sample = slips + rest[:max(N_PER_CLASS - len(slips), 0)]
    return clean[:len(broken_sample)] + broken_sample


def report(sample):
    scored = [s for s in sample if s.get("S") is not None]
    results = {"n_scored": len(scored)}

    # A) clean-vs-broken
    pos = [s["p_clean"] for s in scored if s["is_gold"]]
    neg = [s["p_clean"] for s in scored if not s["is_gold"]]
    a = auc(pos, neg)
    print("=" * 66)
    print(f"A) CLEAN vs BROKEN  (P(no error), {len(pos)} gold / {len(neg)} non-gold)")
    print(f"   AUC = {a:.4f}   [bar 0.70; whole-chain verdict judge: 0.636 contested]")
    results["A_clean_vs_broken"] = {"auc": round(a, 4) if a else None,
                                    "n_gold": len(pos), "n_rest": len(neg)}

    # B) localization on verifiable arithmetic errors
    loc = [s for s in scored if not s["is_gold"] and s["bad_step"]]
    hits = sum(1 for s in loc if s["j_hat"] == s["bad_step"])
    near = sum(1 for s in loc if abs(s["j_hat"] - s["bad_step"]) <= 1)
    chance = (sum(1.0 / (len(s["steps"]) + 1) for s in loc) / len(loc)) if loc else 0.0
    print(f"\nB) LOCALIZATION on verifiable arithmetic slips  (n = {len(loc)})")
    print(f"   exact hit: {hits} ({100*hits/max(len(loc),1):.1f}%)   "
          f"within one step: {near} ({100*near/max(len(loc),1):.1f}%)   "
          f"chance {100*chance:.1f}%")
    results["B_localization"] = {
        "n": len(loc), "exact": hits, "within_one": near,
        "exact_rate": round(hits / max(len(loc), 1), 4),
        "chance_rate": round(chance, 4)}

    # sanity: where does j land on clean chains? should be T+1
    cl = [s for s in scored if s["is_gold"]]
    at_end = sum(1 for s in cl if s["j_hat"] == len(s["steps"]) + 1)
    print(f"\nC) SANITY  clean chains where argmax j = T+1 (no error): "
          f"{at_end}/{len(cl)} ({100*at_end/max(len(cl),1):.1f}%)")
    br = [s for s in scored if not s["is_gold"]]
    br_end = sum(1 for s in br if s["j_hat"] == len(s["steps"]) + 1)
    print(f"   broken chains where argmax j = T+1 (missed the error): "
          f"{br_end}/{len(br)} ({100*br_end/max(len(br),1):.1f}%)")
    results["C_sanity"] = {"clean_at_end": at_end, "n_clean": len(cl),
                           "broken_at_end": br_end, "n_broken": len(br)}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    pools = load_pools(args.root)
    rng = random.Random(SEED)
    sample = build_sample(pools, rng)
    n_gold = sum(s["is_gold"] for s in sample)
    n_bad = sum(1 for s in sample if not s["is_gold"] and s["bad_step"])
    langs = Counter(pools[s["iid"]]["lang"] for s in sample)
    print(f"pools {len(pools)}; sample {len(sample)} chains "
          f"({n_gold} gold / {len(sample)-n_gold} non-gold); "
          f"non-gold with a verifiable arithmetic slip: {n_bad}")
    print(f"languages: {dict(sorted(langs.items()))}")
    print(f"steps per chain: median "
          f"{sorted(len(s['steps']) for s in sample)[len(sample)//2]}")
    print("CAVEAT: Qwen3-14B-Base is a base model; uPRM used an instruct "
          "grader. A null here is partly about base models.")

    if args.dry_run:
        s = next(x for x in sample if not x["is_gold"] and x["bad_step"])
        print(f"\n--- broken chain, first bad step = {s['bad_step']} ---")
        for i, step in enumerate(s["steps"], 1):
            print(f"  [{i}] {step[:100]}")
        print(f"\n--- transcript asserting j = {s['bad_step']} (first 600) ---")
        print(build_transcript(s["question"], s["steps"], s["bad_step"])[:600])
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    plus_id = tokenizer.encode(" +", add_special_tokens=False)[0]
    minus_id = tokenizer.encode(" -", add_special_tokens=False)[0]
    print(f"marker ids: '+' {plus_id}, '-' {minus_id}", flush=True)
    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    # One prompt per (chain, step) under the ALL-CORRECT prefix: that is all
    # Eq. 6 needs, because every hypothesis j shares the same prefix of pluses.
    entries = []
    for index, s in enumerate(sample):
        prefix = SYSTEM + "Problem: " + s["question"].strip()
        for t, step in enumerate(s["steps"], start=1):
            prompt = prefix + f"\n\nStep {t}: {step}\nJudge:"
            n_tokens = len(tokenizer.encode(prompt))
            if n_tokens <= MAX_PROMPT_TOKENS:
                entries.append((n_tokens, index, t, prompt))
            prefix = prefix + f"\n\nStep {t}: {step}\nJudge: +"
    entries.sort(key=lambda e: e[0])
    print(f"scoring {len(entries)} marker slots", flush=True)

    per_chain = defaultdict(dict)
    for start in range(0, len(entries), BATCH_SIZE):
        batch = entries[start:start + BATCH_SIZE]
        enc = tokenizer([p for _, _, _, p in batch], return_tensors="pt",
                        padding=True, pad_to_multiple_of=256).to(model.device)
        with torch.inference_mode():
            logits = model(**enc, use_cache=False, **last_logit).logits
            tail = logits[:, -1, :].float()
            pair = torch.stack([tail[:, plus_id], tail[:, minus_id]], dim=-1)
            logprobs = torch.log_softmax(pair, dim=-1)
        for row, (_, index, t, _) in enumerate(batch):
            per_chain[index][t] = (float(logprobs[row, 0]),
                                   float(logprobs[row, 1]))
        if start % (BATCH_SIZE * 50) == 0:
            print(f"  {start}/{len(entries)}", flush=True)
            torch.cuda.empty_cache()

    for index, s in enumerate(sample):
        slots = per_chain.get(index, {})
        n_steps = len(s["steps"])
        if len(slots) < n_steps:
            continue
        step_logprobs = [slots[t] for t in range(1, n_steps + 1)]
        scores = {j: score_from_logprobs(step_logprobs, j, n_steps)
                  for j in range(1, n_steps + 2)}
        s["S"] = scores
        s["j_hat"] = max(scores, key=lambda j: (scores[j], -j))
        total = max(scores.values())
        lse = total + math.log(sum(math.exp(v - total) for v in scores.values()))
        s["p_clean"] = scores[n_steps + 1] - lse

    results = report(sample)
    (OUT_DIR / "stage32_results.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "stage32_chains.jsonl").open("w", encoding="utf-8") as h:
        for s in sample:
            if s.get("S") is None:
                continue
            h.write(json.dumps({"iid": s["iid"], "is_gold": s["is_gold"],
                                "n_steps": len(s["steps"]),
                                "bad_step": s["bad_step"], "j_hat": s["j_hat"],
                                "p_clean": s["p_clean"]}, ensure_ascii=False) + "\n")
    print("\nwrote stage32_results.json and stage32_chains.jsonl")


if __name__ == "__main__":
    main()
