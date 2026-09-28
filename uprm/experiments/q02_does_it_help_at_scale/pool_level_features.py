"""Stage 38 — IS THERE POOL-LEVEL SIGNAL WHERE CHAIN-LEVEL SIGNAL DIED?

Everything so far scored CHAINS and asked "which answer is right?". That
question is answered at chance: stage30 argmax over candidates 46-52%,
stage36 Arm B broke 3,967 pools to fix 431, stage37 T2 showed per-chain AUC
falling to 0.500 as the wrong answer's local share passes 0.90.

This asks a DIFFERENT question, at a different granularity:
    does a pool that committed a WRONG answer look different, in aggregate,
    from one that committed a CORRECT answer?
Not "pick the right answer" (dead) but "flag this commit as untrustworthy" --
which maps onto the abstention/veto machinery the pipeline already has, and
targets the 3.4% contamination the world-agreement filter cannot see.

FEATURES (per committed pool, all from the stage35 cache, label-blind):
  mean_win     mean p_clean of chains supporting the committed answer
  min_win/max_win, std_win   spread among the winners
  n_win        how many chains support it
  mean_lose    mean p_clean of chains supporting any OTHER answer
  gap          mean_win - mean_lose   (in a clean pool the winner should
               look better than its alternatives; in a fooled pool maybe not)
  margin_best  best winning chain minus best losing chain
  share        the committed answer's local vote share (NOT a PRM feature;
               included as a reference baseline and to test whether the PRM
               features add anything beyond what voting already knows)
  world        cross-language agreement (the existing 0.733 filter)

READOUTS
  1. AUC of each feature alone, separating wrong-commit from correct-commit
     pools. Reference: world-agreement 0.733, share (vote confidence).
  2. The same restricted to pools the world filter CANNOT see (world >= 0.7),
     which is where a new signal would actually be worth something.
  3. A 2-feature combination (gap + mean_win, z-scored) to check whether the
     PRM features carry anything jointly that they lack alone.
  4. Coverage/precision if we ABSTAIN on the lowest-scoring commits -- the
     practical form, compared against abstaining by vote share instead.
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, match
from icm.experiments.q03_does_regeneration_help.regeneration_15lang import weighted_decide
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import World, converge, load

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent


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

    lang_ans = defaultdict(dict)
    for iid, s in final.items():
        if s["answer"] is not None:
            lang_ans[iid.rsplit("_", 1)[-1]][pools[iid]["lang"]] = s["answer"]

    rows = []
    for iid, pool in pools.items():
        committed = final[iid]["answer"]
        if committed is None:
            continue
        win, lose = [], []
        for k, chain in enumerate(pool["chains"]):
            r = scores.get((iid, k))
            if r is None:
                continue
            (win if match(chain["answer"], committed) else lose).append(r["p_clean"])
        if not win:
            continue
        problem = iid.rsplit("_", 1)[-1]
        others = [a for l, a in lang_ans[problem].items()
                  if l != pool["lang"]]
        world_share = (sum(match(committed, a) for a in others) / len(others)
                       if others else 0.0)
        counts = Counter(c["answer"] for c in pool["chains"])
        rows.append({
            "iid": iid, "lang": pool["lang"],
            "correct": match(committed, pool["gold"]),
            "mean_win": st.mean(win),
            "min_win": min(win),
            "max_win": max(win),
            "std_win": st.pstdev(win) if len(win) > 1 else 0.0,
            "n_win": len(win),
            "mean_lose": st.mean(lose) if lose else None,
            "gap": (st.mean(win) - st.mean(lose)) if lose else None,
            "margin_best": (max(win) - max(lose)) if lose else None,
            "share": counts[committed] / len(pool["chains"]) if committed in counts else 0.0,
            "world": world_share,
        })
    good = [r for r in rows if r["correct"]]
    bad = [r for r in rows if not r["correct"]]
    print(f"\ncommitted pools with scores: {len(rows)} "
          f"({len(good)} correct / {len(bad)} WRONG = "
          f"{100*len(bad)/len(rows):.2f}%)")
    results = {"n_pools": len(rows), "n_wrong": len(bad)}

    FEATURES = ("mean_win", "min_win", "max_win", "std_win", "n_win",
                "mean_lose", "gap", "margin_best", "share", "world")

    def table(subset, label):
        print(f"\n=== {label} (n={len(subset)}, "
              f"{sum(1 for r in subset if not r['correct'])} wrong) ===")
        print(f"  {'feature':<14}{'n':>8}{'AUC':>9}   (>0.5 = higher means CORRECT)")
        out = {}
        for f in FEATURES:
            g = [r[f] for r in subset if r["correct"] and r[f] is not None]
            b = [r[f] for r in subset if not r["correct"] and r[f] is not None]
            if not g or not b:
                continue
            a = auc(g, b)
            out[f] = {"n": len(g) + len(b), "auc": round(a, 4)}
            flag = ""
            if f in ("world", "share"):
                flag = "  <- baseline, not a PRM feature"
            print(f"  {f:<14}{len(g)+len(b):>8}{a:>9.4f}{flag}")
        return out

    results["all_pools"] = table(rows, "ALL COMMITTED POOLS")
    hidden = [r for r in rows if r["world"] >= 0.7]
    results["world_ge_0.7"] = table(
        hidden, "POOLS THE WORLD FILTER CANNOT SEE (world >= 0.7)")

    # 3. two-feature combination, z-scored within language
    print("\n=== combined PRM features (z-scored within language) ===")
    for f in ("mean_win", "gap"):
        by_lang = defaultdict(list)
        for r in rows:
            if r[f] is not None:
                by_lang[r["lang"]].append(r[f])
        mu = {l: st.mean(v) for l, v in by_lang.items()}
        sd = {l: max(st.pstdev(v), 1e-9) for l, v in by_lang.items()}
        for r in rows:
            r[f + "_z"] = ((r[f] - mu[r["lang"]]) / sd[r["lang"]]
                           if r[f] is not None else None)
    combo = [r for r in rows if r["mean_win_z"] is not None
             and r["gap_z"] is not None]
    g = [r["mean_win_z"] + r["gap_z"] for r in combo if r["correct"]]
    b = [r["mean_win_z"] + r["gap_z"] for r in combo if not r["correct"]]
    print(f"  mean_win_z + gap_z    n={len(combo):>6}  AUC {auc(g, b):.4f}")
    results["combined"] = {"n": len(combo), "auc": round(auc(g, b), 4)}

    # 4. practical: abstain on the lowest-scoring commits
    print("\n=== abstaining on the least-trusted commits ===")
    n_total = len(pools)
    base_cov = len(rows) / n_total
    base_prec = len(good) / len(rows)
    print(f"  baseline: coverage {base_cov:.4f}  precision {base_prec:.4f}")
    print(f"  {'rule':<26}{'drop':>7}{'coverage':>11}{'precision':>11}")
    frontier = {}
    for name, key in (("PRM mean_win", "mean_win"),
                      ("PRM gap", "gap"),
                      ("vote share (baseline)", "share")):
        usable = [r for r in rows if r[key] is not None]
        usable.sort(key=lambda r: r[key])
        for drop in (0.02, 0.05, 0.10):
            k = int(drop * len(usable))
            kept = usable[k:]
            cov = len(kept) / n_total
            prec = sum(1 for r in kept if r["correct"]) / max(len(kept), 1)
            frontier[f"{key}@{drop}"] = {"coverage": round(cov, 4),
                                         "precision": round(prec, 4)}
            print(f"  {name:<26}{100*drop:>6.0f}%{cov:>11.4f}{prec:>11.4f}")
    results["abstain_frontier"] = frontier

    (OUT_DIR / "stage38_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote stage38_results.json")
    print("DONE")


if __name__ == "__main__":
    main()
