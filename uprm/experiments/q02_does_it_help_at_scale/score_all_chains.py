"""Stage 35 — SCORE EVERY CHAIN AT SCALE (37 langs x 1500), sharded.

WHAT THIS IS: a pure SCORING pass. It attaches Toy 1's step-level score
(p_clean = P(no broken step anywhere), uPRM Eq. 6, INDEPENDENT scoring) to
every chain in two populations, and decides nothing. Decisions run later on
CPU from the cache (stage36), so every combination arm is replayable free.

WHY INDEPENDENT ONLY: Toy 2 (stage33) tested uPRM's joint scoring --
joint 0.848 vs independent 0.847 length-matched (no gain), and poisoning the
neighbours' verdicts dropped it to 0.724 (herding). Joint scoring is not used.

POPULATIONS:
  harvest   chains agreeing with a committed answer -> these become SFT data.
            Stage 34 on phaseA: overall AUC 0.742/0.809, but on the residue
            where the WORLD vote is also fooled (world >= 0.5) it hits
            0.934/0.972 on n=287, vs the old verdict judge's 0.731 on n=47.
  contested chains in pools the loop had to resolve -> lets stage36 test the
            score as a tie-breaker / veto inside the loop. Expectation is
            flat-to-negative: stage30 measured per-pool discrimination at
            46-52% (chance). Run because it is cheap once scored, and because
            a reviewer will ask.

COST (measured, stage34: 9,399 slots in ~3 min of compute = ~52 slots/sec):
  harvest   ~564k chains x ~4.6 steps = ~2.6M slots
  contested ~95k chains x ~4.6 steps  = ~0.44M slots
  ~3.0M slots total / 52 per sec ~= 16 h on one GPU, ~4 h across 4 shards.

SHARDING: --shard i --of n splits by problem id, so a shard is a complete
slice of the world (all languages for its problems) and the caches
concatenate cleanly.

Usage: --self-test | --dry-run [--shard i --of n] | full via
       stage35_score.sbatch (submit once per shard)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17          # patches gate.TIER to 37 languages
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import match
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from ..q01_does_step_scoring_work.toy1_first_broken_step import (CHAIN_MAX_CHARS, MIN_STEPS, SYSTEM,
                               score_from_logprobs, split_steps)

MODEL_DIR = "/scratch/rvyalla/multilingual-icm/assets/models/Qwen3-14B-Base"
DEFAULT_ROOT = ("/scratch/rvyalla/multilingual-icm/results/"
                "qwen3-14b-full37-1500")
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
LANDSLIDE = 0.70
MAX_PROMPT_TOKENS = 3000
BATCH_SIZE = 16
SEED = 20260826


def population(pools, state, contested_ids):
    """Every chain we need a score for, tagged by why we want it.

    harvest   : chain agrees with its pool's COMMITTED answer (enters SFT)
    contested : chain sits in a pool the loop had to resolve
    A chain can be both; it is scored once and tagged with both flags."""
    contested = set(contested_ids)
    rows = []
    for iid in sorted(pools):
        pool = pools[iid]
        committed = state[iid]["answer"]
        for k, chain in enumerate(pool["chains"]):
            if len(chain["text"]) > CHAIN_MAX_CHARS:
                continue
            steps = split_steps(chain["text"])
            if len(steps) < MIN_STEPS:
                continue
            in_harvest = (committed is not None
                          and match(chain["answer"], committed))
            in_contested = iid in contested
            if not (in_harvest or in_contested):
                continue
            rows.append({"iid": iid, "k": k, "lang": pool["lang"],
                         "question": pool["question"], "steps": steps,
                         "answer": chain["answer"],
                         "harvest": in_harvest, "contested": in_contested})
    return rows


def shard_of(iid, n_shards):
    """Split by PROBLEM so each shard holds every language for its problems."""
    return int(iid.rsplit("_", 1)[-1]) % n_shards


def self_test():
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}: got {got!r}, want {want!r}")
        print(f"  {'PASS' if got == want else 'FAIL'}  {name}")

    print("--- sharding ---")
    check("problem 0 -> shard 0", shard_of("train_en_0", 4), 0)
    check("problem 7 -> shard 3", shard_of("train_am_7", 4), 3)
    check("language does not change the shard",
          shard_of("train_en_7", 4) == shard_of("train_sw_7", 4), True)
    ids = [f"train_en_{p}" for p in range(100)]
    counts = Counter(shard_of(i, 4) for i in ids)
    check("shards are balanced", sorted(counts.values()), [25, 25, 25, 25])

    print("--- population tagging ---")
    pools = {
        "train_en_1": {"lang": "en", "gold": "10", "question": "q",
                       "share": 1.0, "plurality": "10",
                       "chains": [{"answer": "10", "text": "s1\ns2\n#### 10"},
                                  {"answer": "20", "text": "s1\ns2\n#### 20"}]},
        "train_de_1": {"lang": "de", "gold": "10", "question": "q",
                       "share": 0.5, "plurality": "10",
                       "chains": [{"answer": "10", "text": "a\nb\n#### 10"},
                                  {"answer": "30", "text": "a\nb\n#### 30"}]},
    }
    state = {"train_en_1": {"answer": "10"}, "train_de_1": {"answer": "10"}}
    rows = population(pools, state, ["train_de_1"])
    by = {(r["iid"], r["k"]): r for r in rows}
    check("committed chain is harvest", by[("train_en_1", 0)]["harvest"], True)
    check("disagreeing chain in a settled pool is skipped",
          ("train_en_1", 1) in by, False)
    check("contested pool keeps its disagreeing chain",
          by[("train_de_1", 1)]["contested"], True)
    check("that chain is not harvest", by[("train_de_1", 1)]["harvest"], False)
    check("a chain can be both",
          (by[("train_de_1", 0)]["harvest"], by[("train_de_1", 0)]["contested"]),
          (True, True))

    print("--- Eq. 6 reuse ---")
    lp = [(-0.1, -2.0), (-0.2, -1.5)]
    check("j=T+1 sums the pluses",
          round(score_from_logprobs(lp, 3, 2), 4), -0.3)

    print()
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        raise SystemExit(1)
    print("ALL SELF-TESTS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--model-dir", default=MODEL_DIR)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    pools = load(args.root)
    world = World(pools)
    state, contested_ids, _ = converge(pools, world)
    rows = population(pools, state, contested_ids)
    mine = [r for r in rows if shard_of(r["iid"], args.of) == args.shard]
    slots = sum(len(r["steps"]) for r in mine)
    print(f"pools {len(pools)}; contested {len(contested_ids)}")
    print(f"chains needing a score: {len(rows)} "
          f"({sum(r['harvest'] for r in rows)} harvest, "
          f"{sum(r['contested'] for r in rows)} contested)")
    print(f"shard {args.shard}/{args.of}: {len(mine)} chains, {slots} slots "
          f"(~{slots/52/3600:.1f} h at 52 slots/s)")

    if args.dry_run:
        r = mine[0]
        print(f"\n--- first chain ({r['lang']}, harvest={r['harvest']}, "
              f"contested={r['contested']}) ---")
        for i, s in enumerate(r["steps"], 1):
            print(f"  [{i}] {s[:90]}")
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

    entries = []
    for index, r in enumerate(mine):
        prefix = SYSTEM + "Problem: " + r["question"].strip()
        for t, step in enumerate(r["steps"], start=1):
            prompt = prefix + f"\n\nStep {t}: {step}\nJudge:"
            n_tokens = len(tokenizer.encode(prompt))
            if n_tokens <= MAX_PROMPT_TOKENS:
                entries.append((n_tokens, index, t, prompt))
            prefix = prefix + f"\n\nStep {t}: {step}\nJudge: +"
    entries.sort(key=lambda e: e[0])
    print(f"scoring {len(entries)} marker slots", flush=True)

    got = defaultdict(dict)
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
            got[index][t] = (float(lp[row, 0]), float(lp[row, 1]))
        if start % (BATCH_SIZE * 500) == 0:
            print(f"  {start}/{len(entries)}", flush=True)
            torch.cuda.empty_cache()

    out = OUT_DIR / f"stage35_scores_{args.shard}of{args.of}.jsonl"
    written = 0
    with out.open("w", encoding="utf-8") as handle:
        for index, r in enumerate(mine):
            slots_r = got.get(index, {})
            n_steps = len(r["steps"])
            if len(slots_r) < n_steps:
                continue
            step_lp = [slots_r[t] for t in range(1, n_steps + 1)]
            scores = {j: score_from_logprobs(step_lp, j, n_steps)
                      for j in range(1, n_steps + 2)}
            top = max(scores.values())
            lse = top + math.log(sum(math.exp(v - top) for v in scores.values()))
            handle.write(json.dumps(
                {"iid": r["iid"], "k": r["k"], "lang": r["lang"],
                 "answer": r["answer"], "n_steps": n_steps,
                 "harvest": r["harvest"], "contested": r["contested"],
                 "p_clean": scores[n_steps + 1] - lse,
                 "j_hat": max(scores, key=lambda j: (scores[j], -j))},
                ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {out} ({written} chains)")


if __name__ == "__main__":
    main()
