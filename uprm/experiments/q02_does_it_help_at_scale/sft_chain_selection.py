"""Stage 39 — CAN THE PRM PICK WHICH CHAIN TO TRAIN ON?

The pipeline commits an answer per pool; several chains support it, and they
all enter the SFT set. Nobody has asked WHICH of them is worth training on.
That question lives in the PRM's strong regime (0.847 on non-landslide wrong
chains) rather than its dead one (0.599 where a wrong answer dominates), so
it is the last untested overlap between what the PRM does well and what the
pipeline needs.

"Best reasoned" has no gold label, so this uses three proxies that do:

  P1 CONTAMINATION AVOIDANCE (hard ground truth).
     In pools that committed a WRONG answer, every supporting chain is a
     wrong training example. But in pools that committed a CORRECT answer,
     some supporting chains still reach it by broken reasoning -- a chain
     with a verifiable false equation that lands on the right number anyway
     (lucky-guess chains, 1.8% of genuine harvest per stage37 T4). A good
     selector should rank those LAST among their pool's winners.
     Metric: AUC within-pool, clean winners vs lucky-guess winners.

  P2 ARITHMETIC VALIDITY (hard ground truth, per chain).
     Same idea globally: does the PRM rank chains with no false equation
     above chains with one, among chains that ALL reach the committed answer?
     This is chain quality with correctness held fixed -- exactly the claim
     that was killed for the old verdict judge (margins tracked formatting,
     not quality). The PRM is a 0.85 signal where that was 0.64, so it earns
     its own test.

  P3 TRANSFER (the practical readout).
     Build three SFT sets of equal size from the same committed pools:
       best     highest-p_clean supporting chain per pool
       random   a random supporting chain per pool (current practice)
       worst    lowest-p_clean supporting chain per pool
     Compare on measurable properties: contamination rate, arithmetic-error
     rate, length, and per-tier composition. A real fine-tune is E6 and is
     out of scope here; this says whether the SETS differ at all, which is
     the precondition for a fine-tune to show anything.

Everything runs on the stage35 cache. CPU only.
"""

from __future__ import annotations

import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from icm.experiments.q02_can_the_model_judge_itself import verdict_judge_gate as gate
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, match
from icm.experiments.q03_does_regeneration_help.regeneration_15lang import weighted_decide
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from ..q01_does_step_scoring_work.toy1_first_broken_step import split_steps
from .error_types import first_bad_step_safe

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
SEED = 20260828


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

    # candidate SFT chains: those supporting the committed answer, with scores
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
            steps = split_steps(chain["text"])
            bad = first_bad_step_safe(steps)
            by_pool[iid].append({
                "k": k, "p": r["p_clean"], "lang": pool["lang"],
                "n_steps": r["n_steps"], "chars": len(chain["text"]),
                "commit_right": commit_right,
                "has_arith_error": bad is not None,
                # a "lucky guess": broken arithmetic, right final answer
                "lucky": bad is not None and commit_right,
            })
    n_chains = sum(len(v) for v in by_pool.values())
    print(f"SFT candidate chains: {n_chains} across {len(by_pool)} pools",
          flush=True)
    results = {"n_pools": len(by_pool), "n_chains": n_chains}

    # ---------------- P2: arithmetic validity, correctness held fixed ------
    print("\n=== P2: rank clean chains above false-equation chains ===")
    print("    (all chains here support the SAME committed answer)")
    for label, keep in (("correct commits only", lambda c: c["commit_right"]),
                        ("wrong commits only", lambda c: not c["commit_right"])):
        good = [c["p"] for v in by_pool.values() for c in v
                if keep(c) and not c["has_arith_error"]]
        bad = [c["p"] for v in by_pool.values() for c in v
               if keep(c) and c["has_arith_error"]]
        a = auc(good, bad) if good and bad else None
        print(f"  {label:<26} {len(good):>7} clean / {len(bad):>5} broken   "
              f"AUC {a if a is not None else float('nan'):.4f}")
        results.setdefault("P2", {})[label] = {
            "n_clean": len(good), "n_broken": len(bad),
            "auc": round(a, 4) if a else None}

    # ---------------- P1: within-pool ranking of lucky guesses -------------
    print("\n=== P1: within a pool, does the lucky-guess chain rank last? ===")
    wins = total = ties = 0
    for iid, chains in by_pool.items():
        if not any(c["lucky"] for c in chains):
            continue
        lucky = [c for c in chains if c["lucky"]]
        clean = [c for c in chains if not c["lucky"]]
        if not clean:
            continue
        total += 1
        best_clean = max(c["p"] for c in clean)
        best_lucky = max(c["p"] for c in lucky)
        if best_clean > best_lucky:
            wins += 1
        elif best_clean == best_lucky:
            ties += 1
    print(f"  pools with both a lucky-guess and a clean winner: {total}")
    print(f"  clean chain scored higher: {wins} "
          f"({100*wins/max(total,1):.1f}%)   ties {ties}   [chance 50%]")
    results["P1"] = {"n_pools": total, "clean_higher": wins, "ties": ties,
                     "rate": round(wins / max(total, 1), 4)}

    # ---------------- P3: do the three SFT sets differ? --------------------
    print("\n=== P3: build best / random / worst SFT sets (1 chain per pool) ===")
    rng = random.Random(SEED)
    sets = {"best": [], "random": [], "worst": []}
    for iid, chains in by_pool.items():
        ordered = sorted(chains, key=lambda c: c["p"])
        sets["worst"].append((iid, ordered[0]))
        sets["best"].append((iid, ordered[-1]))
        sets["random"].append((iid, rng.choice(chains)))
    print(f"  {'set':<9}{'n':>7}{'contaminated':>14}{'false eq':>11}"
          f"{'mean steps':>12}{'mean chars':>12}")
    for name in ("best", "random", "worst"):
        rows = [c for _, c in sets[name]]
        contam = sum(1 for c in rows if not c["commit_right"])
        arith = sum(1 for c in rows if c["has_arith_error"])
        print(f"  {name:<9}{len(rows):>7}{100*contam/len(rows):>13.2f}%"
              f"{100*arith/len(rows):>10.2f}%"
              f"{st.mean(c['n_steps'] for c in rows):>12.2f}"
              f"{st.mean(c['chars'] for c in rows):>12.0f}")
        results.setdefault("P3", {})[name] = {
            "n": len(rows),
            "contaminated_pct": round(100*contam/len(rows), 3),
            "arith_error_pct": round(100*arith/len(rows), 3),
            "mean_steps": round(st.mean(c["n_steps"] for c in rows), 2),
            "mean_chars": round(st.mean(c["chars"] for c in rows), 1)}

    print("\n  per-tier arithmetic-error rate (lower is better):")
    print(f"  {'tier':<7}{'best':>9}{'random':>9}{'worst':>9}")
    for tier in ("high", "mid", "low"):
        line = f"  {tier:<7}"
        for name in ("best", "random", "worst"):
            rows = [c for _, c in sets[name]
                    if gate.TIER.get(c["lang"]) == tier]
            rate = (100 * sum(1 for c in rows if c["has_arith_error"])
                    / max(len(rows), 1))
            line += f"{rate:>8.2f}%"
            results.setdefault("P3_tier", {}).setdefault(tier, {})[name] = round(rate, 3)
        print(line)

    (OUT_DIR / "stage39_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote stage39_results.json")
    print("DONE")


if __name__ == "__main__":
    main()
