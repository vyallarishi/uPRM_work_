"""Stage 33 — uPRM TOY 2: does JOINT scoring add anything beyond context priming?

TOY 1 RESULT (stage32, 700 chains, 15 languages):
  clean-vs-broken AUC 0.835 raw, 0.891 length-matched over 20,782 pairs.
  Length alone scores 0.448 -- the confound runs AGAINST the result.
  First formulation in this project to clear the 0.70 bar.
  BUT localization is at chance: argmax_j hits the verified broken step
  17.6% vs 21.7% chance. The model feels that a chain is broken; it cannot
  say where. p_clean works, argmax_j does not.

uPRM's central claim (Eq. 8) is that scoring ~13 chains in ONE context beats
scoring them individually, because causal attention calibrates the grader on
earlier chains. Their gain from this is most of the paper.

THE HAZARD: stage19 put other languages' chains in context, jumped 0.64 ->
0.92, and the entire gain was answer-copying -- a zero-model baseline scored
0.946. So joint scoring is tested here ONLY alongside controls that would
expose the same failure.

ARMS (identical chains, identical Eq. 6 per-chain scoring, batch of 8):
  independent   Toy 1 replication, one chain per context (anchor)
  joint         8 chains in one context, all-plus prefix, uPRM Eq. 8
  shuffled      8 chains in one context, but the OTHER 7 are from DIFFERENT
                problems -- if joint == shuffled, context is priming the
                marker distribution, not calibrating judgement
  poisoned      the other 7 chains are prefixed with '-' verdicts (asserted
                broken). If the target's score tracks its neighbours'
                asserted verdicts, joint scoring is a herding artifact.

PRE-REGISTERED:
  P1 joint > independent by >= 0.02 AUC (uPRM's claim reproduces here)
  P2 joint - shuffled >= 0.02 (the gain is genuinely about co-scoring these
     chains, not about having any 7 chains in context)
  P3 |joint - poisoned| < 0.05 (the score is not driven by neighbours'
     asserted verdicts)
  Failing P2 or P3 means joint scoring is a context artifact and Toy 3
  (training a PRM against this reward) must not be built.

Usage: --self-test | --dry-run | full via stage33_uprm2.sbatch
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from icm.experiments.q02_can_the_model_judge_itself import verdict_judge_gate as gate
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, load_pools, match
from .toy1_first_broken_step import (CHAIN_MAX_CHARS, MIN_STEPS, SYSTEM,
                               first_bad_step, score_from_logprobs,
                               split_steps)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-phaseA"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
LANGS = ("en", "de", "es", "fr", "ru", "hi", "ar", "ko", "fa",
         "am", "sw", "kn", "eu", "ml", "is")
BATCH_CHAINS = 8           # uPRM packs ~13; 8 keeps us inside 4k context
N_PER_CLASS = 350
MAX_MODEL_LEN = 4096
MAX_PROMPT_TOKENS = 3800
BATCH_SIZE = 8
SEED = 20260824
ARMS = ("independent", "joint", "shuffled", "poisoned")


def chain_block(index, question, steps, verdicts):
    """One graded solution inside a transcript. verdicts[t] in {'+','-',None};
    None means the slot is left open (this is the chain being scored)."""
    lines = [f"Problem {index}: {question.strip()}"]
    for t, step in enumerate(steps, start=1):
        v = verdicts[t - 1] if t - 1 < len(verdicts) else None
        if v is None:
            lines.append(f"Step {t}: {step}\nJudge:")
        else:
            lines.append(f"Step {t}: {step}\nJudge: {v}")
    return "\n".join(lines)


def build_context(neighbours, poisoned=False):
    """Preceding graded solutions. Each is fully marked: all '+' normally,
    all '-' in the poisoned arm."""
    mark = "-" if poisoned else "+"
    blocks = []
    for i, nb in enumerate(neighbours, start=1):
        blocks.append(chain_block(i, nb["question"], nb["steps"],
                                  [mark] * len(nb["steps"])))
    return blocks


def build_slot_prompt(neighbours, target, upto_step, poisoned=False):
    """Context + the target solution with steps 1..upto_step-1 marked '+',
    ending exactly at the marker slot for step upto_step."""
    blocks = build_context(neighbours, poisoned)
    verdicts = ["+"] * (upto_step - 1) + [None]
    blocks.append(chain_block(len(neighbours) + 1, target["question"],
                              target["steps"][:upto_step], verdicts))
    return SYSTEM + "\n\n".join(blocks)


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- chain_block ---")
    b = chain_block(1, "Q?", ["a", "b"], ["+", "-"])
    check("marks both steps", b.count("Judge: "), 2)
    check("keeps order", b.index("Step 1") < b.index("Step 2"), True)
    open_b = chain_block(2, "Q?", ["a"], [None])
    check("open slot has no marker", open_b.endswith("Judge:"), True)

    print("--- build_slot_prompt ---")
    nb = [{"question": "N1", "steps": ["x", "y"]},
          {"question": "N2", "steps": ["z"]}]
    tgt = {"question": "T", "steps": ["p", "q", "r"]}
    p = build_slot_prompt(nb, tgt, 2)
    check("ends at the target slot", p.endswith("Step 2: q\nJudge:"), True)
    check("neighbours all marked plus", p.count("Judge: +"), 4)  # 3 nb + 1 target
    check("target step 3 not shown", "Step 3: r" not in p, True)
    check("system prompt present", p.startswith(SYSTEM.strip()[:20]), True)
    poisoned = build_slot_prompt(nb, tgt, 2, poisoned=True)
    check("poisoned marks neighbours minus", poisoned.count("Judge: -"), 3)
    check("poisoned keeps target prefix plus", poisoned.count("Judge: +"), 1)

    print("--- independent arm is a 0-neighbour joint prompt ---")
    solo = build_slot_prompt([], tgt, 1)
    check("no neighbours", "Problem 2" not in solo, True)
    check("ends at first slot", solo.endswith("Step 1: p\nJudge:"), True)

    print("--- Eq.6 reuse ---")
    lp = [(-0.1, -2.0), (-0.2, -1.5)]
    check("j=T+1 sums pluses",
          round(score_from_logprobs(lp, 3, 2), 4), round(-0.3, 4))

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def build_sample(pools, rng):
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
            row = {"iid": iid, "lang": pool["lang"], "question": pool["question"],
                   "steps": steps, "is_gold": match(chain["answer"], pool["gold"]),
                   "bad_step": first_bad_step(steps)}
            (clean if row["is_gold"] else broken).append(row)
    rng.shuffle(clean)
    rng.shuffle(broken)
    n = min(N_PER_CLASS, len(clean), len(broken))
    sample = clean[:n] + broken[:n]
    rng.shuffle(sample)
    return sample


def report(sample):
    results = {}
    print(f"{'arm':<14}{'n':>6}{'AUC':>9}{'len-matched':>13}")
    for arm in ARMS:
        scored = [s for s in sample if s.get(f"p_clean_{arm}") is not None]
        if not scored:
            continue
        key = f"p_clean_{arm}"
        g = [s[key] for s in scored if s["is_gold"]]
        b = [s[key] for s in scored if not s["is_gold"]]
        a = auc(g, b)
        strata = defaultdict(list)
        for s in scored:
            strata[len(s["steps"])].append(s)
        pairs = conc = 0
        for sub in strata.values():
            gg = [s[key] for s in sub if s["is_gold"]]
            bb = [s[key] for s in sub if not s["is_gold"]]
            if gg and bb:
                pairs += len(gg) * len(bb)
                conc += auc(gg, bb) * len(gg) * len(bb)
        lm = conc / pairs if pairs else None
        results[arm] = {"n": len(scored), "auc": round(a, 4) if a else None,
                        "auc_length_matched": round(lm, 4) if lm else None,
                        "n_gold": len(g), "n_rest": len(b)}
        print(f"{arm:<14}{len(scored):>6}{a:>9.4f}"
              f"{lm if lm else float('nan'):>13.4f}")

    def gap(x, y):
        ax = results.get(x, {}).get("auc")
        ay = results.get(y, {}).get("auc")
        return None if ax is None or ay is None else round(ax - ay, 4)

    p1, p2, p3 = gap("joint", "independent"), gap("joint", "shuffled"), gap("joint", "poisoned")
    print(f"\nP1 joint - independent = {p1}  [need >= +0.02]  "
          f"{'PASS' if p1 and p1 >= 0.02 else 'FAIL'}")
    print(f"P2 joint - shuffled    = {p2}  [need >= +0.02]  "
          f"{'PASS' if p2 and p2 >= 0.02 else 'FAIL'}")
    print(f"P3 |joint - poisoned|  = {abs(p3) if p3 is not None else None}  "
          f"[need < 0.05]  {'PASS' if p3 is not None and abs(p3) < 0.05 else 'FAIL'}")
    results["prereg"] = {"P1_joint_vs_independent": p1,
                         "P2_joint_vs_shuffled": p2,
                         "P3_joint_vs_poisoned": p3}
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
    by_problem = defaultdict(list)
    for s in sample:
        by_problem[s["iid"].rsplit("_", 1)[-1]].append(s)

    # neighbour sets, fixed per target so all arms see the same partners
    for index, s in enumerate(sample):
        pool_mates = [o for o in by_problem[s["iid"].rsplit("_", 1)[-1]]
                      if o is not s]
        r = random.Random((SEED, index).__repr__())
        r.shuffle(pool_mates)
        same = pool_mates[:BATCH_CHAINS - 1]
        if len(same) < BATCH_CHAINS - 1:      # top up from anywhere
            others = [o for o in sample if o is not s and o not in same]
            r.shuffle(others)
            same += others[:BATCH_CHAINS - 1 - len(same)]
        s["neighbours_same"] = same
        problem = s["iid"].rsplit("_", 1)[-1]
        diff = [o for o in sample
                if o is not s and o["iid"].rsplit("_", 1)[-1] != problem]
        r.shuffle(diff)
        s["neighbours_diff"] = diff[:BATCH_CHAINS - 1]

    print(f"pools {len(pools)}; sample {len(sample)} "
          f"({sum(x['is_gold'] for x in sample)} clean / "
          f"{sum(not x['is_gold'] for x in sample)} broken); "
          f"batch {BATCH_CHAINS} chains/context")
    print(f"languages: {dict(sorted(Counter(s['lang'] for s in sample).items()))}")

    if args.dry_run:
        s = sample[0]
        p = build_slot_prompt(s["neighbours_same"], s, 1)
        print(f"\n--- joint prompt, first slot ({len(p)} chars, last 500) ---")
        print(p[-500:])
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    plus_id = tokenizer.encode(" +", add_special_tokens=False)[0]
    minus_id = tokenizer.encode(" -", add_special_tokens=False)[0]
    print("loading model in bf16", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    import inspect
    last_logit = ({"logits_to_keep": 1}
                  if "logits_to_keep" in inspect.signature(model.forward).parameters
                  else {})

    for arm in ARMS:
        entries = []
        for index, s in enumerate(sample):
            if arm == "independent":
                nb, poisoned = [], False
            elif arm == "joint":
                nb, poisoned = s["neighbours_same"], False
            elif arm == "shuffled":
                nb, poisoned = s["neighbours_diff"], False
            else:
                nb, poisoned = s["neighbours_same"], True
            for t in range(1, len(s["steps"]) + 1):
                prompt = build_slot_prompt(nb, s, t, poisoned)
                n_tokens = len(tokenizer.encode(prompt))
                if n_tokens <= MAX_PROMPT_TOKENS:
                    entries.append((n_tokens, index, t, prompt))
        entries.sort(key=lambda e: e[0])
        print(f"[{arm}] scoring {len(entries)} slots", flush=True)
        slots = defaultdict(dict)
        for start in range(0, len(entries), BATCH_SIZE):
            batch = entries[start:start + BATCH_SIZE]
            enc = tokenizer([p for _, _, _, p in batch], return_tensors="pt",
                            padding=True, pad_to_multiple_of=256).to(model.device)
            with torch.inference_mode():
                logits = model(**enc, use_cache=False, **last_logit).logits
                tail = logits[:, -1, :].float()
                pair = torch.stack([tail[:, plus_id], tail[:, minus_id]], dim=-1)
                lp = torch.log_softmax(pair, dim=-1)
            for row, (_, index, t, _) in enumerate(batch):
                slots[index][t] = (float(lp[row, 0]), float(lp[row, 1]))
            if start % (BATCH_SIZE * 100) == 0:
                print(f"  {start}/{len(entries)}", flush=True)
                torch.cuda.empty_cache()
        for index, s in enumerate(sample):
            got = slots.get(index, {})
            n_steps = len(s["steps"])
            if len(got) < n_steps:
                continue
            step_lp = [got[t] for t in range(1, n_steps + 1)]
            scores = {j: score_from_logprobs(step_lp, j, n_steps)
                      for j in range(1, n_steps + 2)}
            top = max(scores.values())
            lse = top + math.log(sum(math.exp(v - top) for v in scores.values()))
            s[f"p_clean_{arm}"] = scores[n_steps + 1] - lse
            s[f"j_hat_{arm}"] = max(scores, key=lambda j: (scores[j], -j))

    results = report(sample)
    (OUT_DIR / "stage33_results.json").write_text(json.dumps(results, indent=2))
    with (OUT_DIR / "stage33_chains.jsonl").open("w", encoding="utf-8") as h:
        for s in sample:
            h.write(json.dumps(
                {k: v for k, v in s.items()
                 if k not in ("steps", "question", "neighbours_same",
                              "neighbours_diff")}, ensure_ascii=False) + "\n")
    print("\nwrote stage33_results.json and stage33_chains.jsonl")


if __name__ == "__main__":
    main()
