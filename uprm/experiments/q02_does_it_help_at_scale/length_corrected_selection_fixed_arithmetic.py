"""Stage 40 — DOES LENGTH-CORRECTING THE SELECTOR RESCUE IT?

Stage 39 found a real but narrow effect (P1: within pools containing both a
lucky-guess chain and a properly-reasoned one, the PRM ranked the clean chain
higher 67.4% of 2,619 pools, chance 50%) sitting inside a selector that is
globally dominated by LENGTH: picking argmax p_clean picks 3.96-step chains
against 5.56 for argmin, and the "worst" set ends up with FEWER false
equations than the "best" set.

That length coupling is expected: p_clean sums one log P(+) per step, so
longer chains score lower mechanically. In Toy 1 length ran AGAINST the
signal and correcting for it improved AUC 0.835 -> 0.891. This asks whether
the same correction rescues chain SELECTION.

VARIANTS (all label-blind, computable from the stage35 cache):
  raw        p_clean as-is (stage39's selector)
  per_step   p_clean / n_steps
  resid      p_clean minus a global linear fit on n_steps
  within     rank within the pool's own step-count stratum, then p_clean
             (ties broken by p_clean) -- selection restricted to comparable
             chains

READOUTS
  A P1 rerun (lucky-guess vs clean, within pool) for each variant
  B P2 rerun (clean vs false-equation, global) for each variant
  C P3 rerun: SFT set composition for each variant -- contamination, false
    equations, mean steps, mean chars. The length effect should visibly
    shrink if the correction works.
  D what P1's win is actually worth: restricted to the 2,619 pools where a
    lucky-guess chain exists, how does each variant change the set?
"""

from __future__ import annotations

import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from icm.experiments.q02_can_the_model_judge_itself import verdict_judge_gate as gate
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, match
from icm.experiments.q03_does_regeneration_help.regeneration_15lang import weighted_decide
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from ..q01_does_step_scoring_work.toy1_first_broken_step import split_steps
from icm.experiments.q07_audits.fix_arithmetic_precedence import first_bad_step as first_bad_step_safe

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
SEED = 20260828
VARIANTS = ("raw", "per_step", "resid", "within")


def main():
    scores = {}
    for shard in range(4):
        for line in (OUT_DIR / f"stage35_scores_{shard}of4.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            scores[(r["iid"], r["k"])] = r
    print(f"loaded {len(scores)} scored chains", flush=True)

    pools = load(ROOT)
    world = World(pools)
    state, contested_ids, osc = converge(pools, world)
    regen = {}
    path = OUT_DIR / "stage25_regen.jsonl"
    if path.exists():
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            regen[r["iid"]] = r["answers"]
    final = {i: dict(s) for i, s in state.items()}
    for iid in contested_ids:
        final[iid]["answer"] = weighted_decide(
            pools[iid], world.context(final, iid), regen.get(iid, []), 0.5,
            iid in osc)
    print("converged", flush=True)

    by_pool = defaultdict(list)
    for iid, pool in pools.items():
        committed = final[iid]["answer"]
        if committed is None:
            continue
        commit_right = match(committed, pool["gold"])
        for k, chain in enumerate(pool["chains"]):
            r = scores.get((iid, k))
            if r is None or not match(chain["answer"], committed):
                continue
            bad = first_bad_step_safe(split_steps(chain["text"]))
            by_pool[iid].append({
                "k": k, "raw": r["p_clean"], "lang": pool["lang"],
                "n_steps": r["n_steps"], "chars": len(chain["text"]),
                "commit_right": commit_right,
                "has_arith_error": bad is not None,
                "lucky": bad is not None and commit_right})
    allc = [c for v in by_pool.values() for c in v]
    print(f"SFT candidates: {len(allc)} across {len(by_pool)} pools", flush=True)

    # ---- build the length-corrected variants
    xs = [c["n_steps"] for c in allc]
    ys = [c["raw"] for c in allc]
    mx, my = st.mean(xs), st.mean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0
    print(f"length slope: {slope:+.4f} per step")
    for c in allc:
        c["per_step"] = c["raw"] / max(c["n_steps"], 1)
        c["resid"] = c["raw"] - (my + slope * (c["n_steps"] - mx))
    # 'within': prefer chains at the pool's modal step count, then p_clean
    for iid, chains in by_pool.items():
        counts = defaultdict(int)
        for c in chains:
            counts[c["n_steps"]] += 1
        modal = max(counts, key=lambda n: (counts[n], -n))
        for c in chains:
            c["within"] = (0 if c["n_steps"] == modal else -1, c["raw"])
    results = {"length_slope": round(slope, 4), "n_chains": len(allc),
               "n_pools": len(by_pool)}

    def key(c, v):
        return c[v]

    # ---- A: P1 rerun
    print("\n=== A: P1 -- within pool, clean chain vs lucky-guess chain ===")
    print(f"  {'variant':<10}{'pools':>8}{'clean higher':>14}{'rate':>9}  [chance 50%]")
    for v in VARIANTS:
        wins = total = 0
        for iid, chains in by_pool.items():
            lucky = [c for c in chains if c["lucky"]]
            clean = [c for c in chains if not c["lucky"]]
            if not lucky or not clean:
                continue
            total += 1
            if max(key(c, v) for c in clean) > max(key(c, v) for c in lucky):
                wins += 1
        print(f"  {v:<10}{total:>8}{wins:>14}{100*wins/max(total,1):>8.1f}%")
        results.setdefault("A_P1", {})[v] = {
            "n_pools": total, "wins": wins,
            "rate": round(wins / max(total, 1), 4)}

    # ---- B: P2 rerun (global, scalar variants only)
    print("\n=== B: P2 -- global, clean vs false-equation chains ===")
    print(f"  {'variant':<10}{'AUC':>9}")
    for v in ("raw", "per_step", "resid"):
        good = [c[v] for c in allc if not c["has_arith_error"]]
        bad = [c[v] for c in allc if c["has_arith_error"]]
        a = auc(good, bad)
        print(f"  {v:<10}{a:>9.4f}")
        results.setdefault("B_P2", {})[v] = round(a, 4)

    # ---- C: P3 rerun
    print("\n=== C: SFT set composition (1 chain per pool) ===")
    rng = random.Random(SEED)
    print(f"  {'selector':<14}{'contam':>9}{'false eq':>10}{'steps':>8}{'chars':>8}")
    rows_random = [rng.choice(v) for v in by_pool.values()]
    def show(name, rows):
        contam = 100 * sum(1 for c in rows if not c["commit_right"]) / len(rows)
        arith = 100 * sum(1 for c in rows if c["has_arith_error"]) / len(rows)
        steps = st.mean(c["n_steps"] for c in rows)
        chars = st.mean(c["chars"] for c in rows)
        print(f"  {name:<14}{contam:>8.2f}%{arith:>9.2f}%{steps:>8.2f}{chars:>8.0f}")
        return {"contaminated_pct": round(contam, 3),
                "arith_error_pct": round(arith, 3),
                "mean_steps": round(steps, 2), "mean_chars": round(chars, 1)}
    results.setdefault("C_sets", {})["random"] = show("random", rows_random)
    for v in VARIANTS:
        rows = [max(chains, key=lambda c: key(c, v))
                for chains in by_pool.values()]
        results["C_sets"][f"best_{v}"] = show(f"best[{v}]", rows)

    # ---- D: what is P1's win worth, restricted to the pools where it applies
    print("\n=== D: restricted to pools containing a lucky-guess chain ===")
    lucky_pools = {iid: v for iid, v in by_pool.items()
                   if any(c["lucky"] for c in v) and any(not c["lucky"] for c in v)}
    print(f"  {len(lucky_pools)} pools ({100*len(lucky_pools)/len(by_pool):.1f}% "
          f"of all committed pools)")
    print(f"  {'selector':<14}{'picks a lucky-guess chain':>28}")
    for v in VARIANTS:
        picked = sum(1 for chains in lucky_pools.values()
                     if max(chains, key=lambda c: key(c, v))["lucky"])
        print(f"  {v:<14}{picked:>20} ({100*picked/len(lucky_pools):.1f}%)")
        results.setdefault("D_lucky", {})[v] = {
            "picked_lucky": picked,
            "pct": round(100 * picked / max(len(lucky_pools), 1), 2)}
    rand_lucky = sum(1 for chains in lucky_pools.values()
                     if rng.choice(chains)["lucky"])
    print(f"  {'random':<14}{rand_lucky:>20} "
          f"({100*rand_lucky/len(lucky_pools):.1f}%)")
    results["D_lucky"]["random"] = {
        "picked_lucky": rand_lucky,
        "pct": round(100 * rand_lucky / max(len(lucky_pools), 1), 2)}

    (OUT_DIR / "stage40R_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote stage40R_results.json")
    print("DONE")


if __name__ == "__main__":
    main()
