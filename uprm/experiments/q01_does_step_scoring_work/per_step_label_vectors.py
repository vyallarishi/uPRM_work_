"""Stage 44 — PER-STEP +/- LABEL VECTORS (the user's design, not uPRM's j).

THE DIFFERENCE FROM STAGE 32/35, stated precisely:
  stage32/35 store ONE ordinal per chain -- j, the position of the first
  error (uPRM Eq. 6). Everything after j is assumed wrong; the chain's
  label is a single number.
  THIS FILE stores a VECTOR: one +/- decision per step, each read from the
  model's renormalized P(+) vs P(-) at that step's marker slot. A 5-step
  chain yields 5 latent labels, not 1. A chain may read (+, -, +, +):
  the absorbing assumption is DROPPED.

WHY DROP IT (measured, not assumed): stage42 tested uPRM's own
absorbing-error system prompt ("once a step is incorrect, consider all
subsequent steps incorrect") against stage32's prompt on identical chains.
Localization got WORSE: lift -4.1pts vs -0.5pts, P1 FAILED. The absorbing
assumption is not earning its place on this base model, so the free-label
parameterization is the better-supported one here.

WHY THIS NEEDS A RESCORE AT ALL: stage35 already COMPUTES these exact
per-step (log p+, log p-) pairs (stage35_score_all.py:227) and then
DISCARDS them, writing only p_clean and j_hat. The numbers we need were
produced and thrown away. This file is that pass again, saving the vector.

PREFIX ARMS (the design question this file settles):
  allplus   every earlier step shown marked '+', regardless of what the
            model said about it. What stage32/35 do; what uPRM Eq. 6
            requires. Fully batchable: one forward pass per (chain, step),
            all independent.
  selfcons  every earlier step shown with the marker the MODEL assigned
            (greedy argmax at that slot), fed forward. Closer to a real
            PRM transcript, but step t's prompt depends on step t-1's
            outcome, so a chain must be scored sequentially. Run on a
            SUBSAMPLE only.

The arms answer: does the prefix choice change the labels? If they mostly
agree on the overlap, allplus is justified for the full population and the
cheap batched pass stands. If they diverge, every stage32/35 number
inherits an assumption that does real work, which is itself a finding.

PRE-REGISTERED (gold is NOT used anywhere in this file -- it is a pure
scoring pass; gold enters only in later analysis stages):
  P1 AGREEMENT: on the subsample scored both ways, per-step label
     agreement (allplus vs selfcons) >= 0.90. Below that, the all-plus
     prefix is doing real work and stage35's cached j_hat inherits it.
  P2 NON-DEGENERACY: the free-label vectors are not simply absorbing.
     Report the fraction of chains containing a '-' followed by a later
     '+' ("recovery"). If that fraction is ~0, the model is behaving as
     if absorbing anyway and the free parameterization buys nothing.
  P3 COST: allplus slot count matches stage35's (~3.0M), confirming this
     is the same work with a different write.

OUTPUT, one line per chain:
  {"iid","k","lang","answer","n_steps","harvest","contested",
   "lp": [[logp_plus, logp_minus], ...],   # one pair per step
   "labels": "+-++",                        # argmax per step
   "first_minus": 3 | null,                 # uPRM j recoverable from this
   "p_clean_uprm": float}                   # Eq. 6 value, for continuity

Storing lp (not just labels) means every downstream energy -- hard
assignment, soft/probabilistic, or a re-derived j -- runs off ONE pass.

COST: allplus ~3.0M slots ~= 4h across 4 shards (stage35 measured
~52 slots/sec/GPU). selfcons subsample 20k chains ~= 92k slots but
sequential per chain; batched ACROSS chains at the same step index, so
~= 1 GPU-hour.

Usage: --self-test | --dry-run | --arm allplus --shard i --of n
       | --arm selfcons --subsample 20000
       via stage44_perstep.sbatch
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17            # patches gate.TIER to 37 langs
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import load_pools
from .toy1_first_broken_step import SYSTEM, score_from_logprobs
from .paper_faithful_rerun import split_steps_fallback

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = ("/scratch/rvyalla/multilingual-icm/results/"
                "qwen3-14b-full37-1500")
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
MAX_STEPS = 12
MIN_STEPS = 2
CHAIN_MAX_CHARS = 1400
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 16
SEED = 20260830
SUBSAMPLE = 20000


def build_rows(root, splitter):
    """Every chain, with its steps. No gold, no scoring."""
    pools = load_pools(root)
    rows = []
    for iid in sorted(pools):
        pool = pools[iid]
        for k, chain in enumerate(pool["chains"]):
            if len(chain["text"]) > CHAIN_MAX_CHARS:
                continue
            steps = splitter(chain["text"])[:MAX_STEPS]
            if len(steps) < MIN_STEPS:
                continue
            rows.append({"iid": iid, "k": k, "lang": pool["lang"],
                         "answer": chain["answer"], "question": pool["question"],
                         "steps": steps})
    return rows, pools


def allplus_prompts(question, steps):
    """One prompt per step, every earlier marker '+'. Order-independent."""
    out = []
    prefix = SYSTEM + "Problem: " + question.strip()
    for t, step in enumerate(steps, start=1):
        out.append(prefix + f"\n\nStep {t}: {step}\nJudge:")
        prefix = prefix + f"\n\nStep {t}: {step}\nJudge: +"
    return out


def selfcons_prompt(question, steps, markers_so_far):
    """Prompt for the NEXT unscored step, given the markers the model has
    already chosen for the earlier ones. markers_so_far is a list of ' +'
    / ' -' of length t-1; the returned prompt reads step t's slot."""
    prefix = SYSTEM + "Problem: " + question.strip()
    for i, step in enumerate(steps[:len(markers_so_far)]):
        prefix = prefix + f"\n\nStep {i+1}: {step}\nJudge:{markers_so_far[i]}"
    t = len(markers_so_far) + 1
    return prefix + f"\n\nStep {t}: {steps[t-1]}\nJudge:"


def labels_from_lp(lp):
    """argmax per step -> '+'/'-' string."""
    return "".join("+" if p >= m else "-" for p, m in lp)


def first_minus(labels):
    i = labels.find("-")
    return (i + 1) if i >= 0 else None


def uprm_p_clean(lp):
    """Eq. 6's P(no error), recomputed from the same vector, so this file's
    output stays comparable to stage32/35's cached p_clean."""
    n = len(lp)
    scores = {j: score_from_logprobs(lp, j, n) for j in range(1, n + 2)}
    top = max(scores.values())
    lse = top + math.log(sum(math.exp(v - top) for v in scores.values()))
    return scores[n + 1] - lse


def has_recovery(labels):
    """Does a '-' get followed by a later '+'? (violates absorbing)"""
    i = labels.find("-")
    return i >= 0 and "+" in labels[i + 1:]


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- labels_from_lp ---")
    check("all plus", labels_from_lp([(-0.1, -2.0), (-0.2, -3.0)]), "++")
    check("all minus", labels_from_lp([(-2.0, -0.1), (-3.0, -0.2)]), "--")
    check("mixed", labels_from_lp([(-0.1, -2.0), (-3.0, -0.2)]), "+-")
    check("tie goes to plus", labels_from_lp([(-1.0, -1.0)]), "+")

    print("--- first_minus (uPRM j is recoverable) ---")
    check("no minus -> None", first_minus("++++"), None)
    check("minus at 1", first_minus("-+++"), 1)
    check("minus at 3", first_minus("++-+"), 3)

    print("--- has_recovery: the non-absorbing case ---")
    check("clean chain: no recovery", has_recovery("++++"), False)
    check("absorbing chain: no recovery", has_recovery("++--"), False)
    check("recovery present", has_recovery("+-++"), True)
    check("recovery at the end", has_recovery("+--+"), True)

    print("--- allplus prompts are order-independent & all-plus ---")
    steps = ["one", "two", "three"]
    ps = allplus_prompts("Q?", steps)
    check("one prompt per step", len(ps), 3)
    check("first ends at its marker",
          ps[0].endswith("Step 1: one\nJudge:"), True)
    check("second carries a '+' for step 1",
          "Step 1: one\nJudge: +" in ps[1], True)
    check("no '-' ever appears in an allplus prompt",
          any("Judge: -" in p for p in ps), False)
    check("third carries '+' for steps 1 and 2",
          ps[2].count("Judge: +"), 2)

    print("--- allplus matches stage35's construction byte-for-byte ---")
    # stage35_score_all.py:200-207 builds exactly this. If this drifts, the
    # rescore is not comparable to the cached p_clean/j_hat.
    ref, prefix = [], SYSTEM + "Problem: " + "Q?"
    for t, step in enumerate(steps, start=1):
        ref.append(prefix + f"\n\nStep {t}: {step}\nJudge:")
        prefix = prefix + f"\n\nStep {t}: {step}\nJudge: +"
    check("identical to stage35's prompts", ps, ref)

    print("--- selfcons prompts follow the model's own markers ---")
    p0 = selfcons_prompt("Q?", steps, [])
    check("first slot identical to allplus", p0, ps[0])
    p1 = selfcons_prompt("Q?", steps, [" -"])
    check("second slot carries the '-' the model chose",
          "Step 1: one\nJudge: -" in p1, True)
    check("second slot reads step 2",
          p1.endswith("Step 2: two\nJudge:"), True)
    p2 = selfcons_prompt("Q?", steps, [" -", " +"])
    check("third slot carries both prior markers",
          "Judge: -" in p2 and "Judge: +" in p2, True)
    check("selfcons == allplus when all markers are '+'",
          selfcons_prompt("Q?", steps, [" +", " +"]), ps[2])

    print("--- uPRM p_clean recomputes stage32's anchor values ---")
    lp = [(-0.1, -2.0), (-0.2, -1.5), (-0.3, -1.0)]
    check("j=1 term", round(score_from_logprobs(lp, 1, 3), 4), -2.0)
    check("j=T+1 term", round(score_from_logprobs(lp, 4, 3), 4), -0.6)
    pc = uprm_p_clean(lp)
    check("p_clean is a log-probability (<= 0)", pc <= 0.0, True)
    check("p_clean finite", math.isfinite(pc), True)

    print("--- vector is strictly richer than j ---")
    # two chains with the SAME uPRM j but different label vectors
    a, b = "+-++", "+---"
    check("same first_minus", first_minus(a), first_minus(b))
    check("different vectors", a == b, False)
    check("only one shows recovery",
          (has_recovery(a), has_recovery(b)), (True, False))

    print()
    if failures:
        print(f"{len(failures)} FAILURES:")
        for f in failures:
            print(f"  {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arm", choices=("allplus", "selfcons"),
                        default="allplus")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--subsample", type=int, default=SUBSAMPLE)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    rows, _ = build_rows(args.root, split_steps_fallback)
    # shard by problem id so a shard is a complete slice of the world
    if args.of > 1:
        rows = [r for r in rows
                if hash(r["iid"].rsplit("_", 1)[-1]) % args.of == args.shard]
    if args.arm == "selfcons":
        rng = random.Random(SEED)
        rng.shuffle(rows)
        rows = rows[:args.subsample]

    n_slots = sum(len(r["steps"]) for r in rows)
    print(f"arm {args.arm}; chains {len(rows)}; marker slots {n_slots}")
    print(f"median steps "
          f"{sorted(len(r['steps']) for r in rows)[len(rows)//2] if rows else 0}")
    print("NOTE: gold is not loaded or used anywhere in this file.")

    if args.dry_run:
        r = rows[0]
        print(f"\n--- {r['iid']} k={r['k']} ({len(r['steps'])} steps) ---")
        for i, s in enumerate(r["steps"], 1):
            print(f"  [{i}] {s[:90]}")
        print(f"\n--- allplus prompt for step {len(r['steps'])} (last 400) ---")
        print(allplus_prompts(r["question"], r["steps"])[-1][-400:])
        print(f"\n--- selfcons prompt, model said '-' at step 1 (last 400) ---")
        print(selfcons_prompt(r["question"], r["steps"],
                              [" -"] + [" +"] * (len(r["steps"]) - 2))[-400:])
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

    def score_batch(prompts):
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        pad_to_multiple_of=256).to(model.device)
        with torch.inference_mode():
            logits = model(**enc, use_cache=False, **last_logit).logits
            tail = logits[:, -1, :].float()
            pair = torch.stack([tail[:, plus_id], tail[:, minus_id]], dim=-1)
            lp = torch.log_softmax(pair, dim=-1)
        return [(float(lp[i, 0]), float(lp[i, 1])) for i in range(len(prompts))]

    per_chain = defaultdict(dict)

    if args.arm == "allplus":
        entries = []
        for index, r in enumerate(rows):
            for t, prompt in enumerate(allplus_prompts(r["question"],
                                                       r["steps"]), start=1):
                if len(tokenizer.encode(prompt)) <= MAX_PROMPT_TOKENS:
                    entries.append((len(prompt), index, t, prompt))
        entries.sort(key=lambda e: e[0])
        print(f"scoring {len(entries)} slots", flush=True)
        for start in range(0, len(entries), BATCH_SIZE):
            batch = entries[start:start + BATCH_SIZE]
            got = score_batch([p for _, _, _, p in batch])
            for (_, index, t, _), pair in zip(batch, got):
                per_chain[index][t] = pair
            if start % (BATCH_SIZE * 500) == 0:
                print(f"  {start}/{len(entries)}", flush=True)
                torch.cuda.empty_cache()
    else:
        # Sequential within a chain, but batched ACROSS chains at the same
        # step index: every chain still needing step t is scored together.
        markers = {i: [] for i in range(len(rows))}
        alive = [i for i in range(len(rows))]
        t = 1
        while alive:
            batchable = [i for i in alive if len(rows[i]["steps"]) >= t]
            if not batchable:
                break
            print(f"  step {t}: {len(batchable)} chains", flush=True)
            for start in range(0, len(batchable), BATCH_SIZE):
                chunk = batchable[start:start + BATCH_SIZE]
                prompts, keep = [], []
                for i in chunk:
                    p = selfcons_prompt(rows[i]["question"], rows[i]["steps"],
                                        markers[i])
                    if len(tokenizer.encode(p)) <= MAX_PROMPT_TOKENS:
                        prompts.append(p)
                        keep.append(i)
                if not prompts:
                    continue
                got = score_batch(prompts)
                for i, pair in zip(keep, got):
                    per_chain[i][t] = pair
                    markers[i].append(" +" if pair[0] >= pair[1] else " -")
            alive = [i for i in batchable if len(rows[i]["steps"]) > t]
            t += 1
            torch.cuda.empty_cache()

    tag = (f"_{args.arm}" +
           (f"_{args.shard}of{args.of}" if args.of > 1 else ""))
    out = OUT_DIR / f"stage44_perstep{tag}.jsonl"
    written = 0
    stats = Counter()
    with out.open("w", encoding="utf-8") as handle:
        for index, r in enumerate(rows):
            slots = per_chain.get(index, {})
            n = len(r["steps"])
            if len(slots) < n:
                continue
            lp = [slots[t] for t in range(1, n + 1)]
            labels = labels_from_lp(lp)
            stats["chains"] += 1
            stats["recovery"] += has_recovery(labels)
            stats["all_plus"] += (labels.count("-") == 0)
            handle.write(json.dumps({
                "iid": r["iid"], "k": r["k"], "lang": r["lang"],
                "answer": r["answer"], "n_steps": n,
                "lp": [[round(a, 6), round(b, 6)] for a, b in lp],
                "labels": labels,
                "first_minus": first_minus(labels),
                "p_clean_uprm": round(uprm_p_clean(lp), 6),
            }, ensure_ascii=False) + "\n")
            written += 1
    c = max(stats["chains"], 1)
    print(f"\nwrote {out} ({written} chains)")
    print(f"P2 NON-DEGENERACY: {stats['recovery']} chains "
          f"({100*stats['recovery']/c:.1f}%) contain a '-' followed by a "
          f"later '+' (absorbing would give 0%)")
    print(f"  all-plus (no '-' anywhere): {stats['all_plus']} "
          f"({100*stats['all_plus']/c:.1f}%)")


if __name__ == "__main__":
    main()
