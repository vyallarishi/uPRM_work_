"""Stage 36 — DECIDE using the stage35 scores (CPU, no GPU).

Stage 35 scored 344,383 chains at 37 langs x 1500 with Toy 1's step-level
p_clean and decided nothing. This replays every way of USING that score
against the shipped pipeline, from the cache, in minutes.

ARMS
  0 baseline        shipped pipeline: converge -> commit. No score used.
  A harvest clean   drop lowest-p_clean chains from the SFT set. Two variants:
                    globally, and only inside the residue where the WORLD vote
                    is also fooled -- the place stage34 measured 0.934 AUC
                    (n=287) against the verdict judge's 0.731 (n=47).
  B in-loop tiebreak use p_clean to break ties among a contested pool's
                    candidate answers (the old judge's role).
  C uPRM alone      ignore voting entirely: each pool commits the answer whose
                    best chain scores highest.

PRE-REGISTERED EXPECTATIONS (stated before running, from prior measurements):
  A should help; it is the only arm with supporting evidence (stage34 H2).
  B and C should be flat-to-negative: stage30 measured per-pool discrimination
    at 46-52% (chance). They are run because they are free and because a
    reviewer will ask for the uPRM-alone baseline.
Gold is used only in evaluation.
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from icm.experiments.q02_can_the_model_judge_itself import verdict_judge_gate as gate
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, match
from icm.experiments.q03_does_regeneration_help.regeneration_15lang import weighted_decide
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from icm.experiments.q01_can_voting_label_data.consensus_loop_15lang import metrics

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
WORLD_STRENGTH = 0.70


def load_scores():
    scores = {}
    for shard in range(4):
        path = OUT_DIR / f"stage35_scores_{shard}of4.jsonl"
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            scores[(r["iid"], r["k"])] = r
    return scores


def main():
    scores = load_scores()
    print(f"loaded {len(scores)} scored chains")
    pools = load(ROOT)
    world = World(pools)
    state, contested_ids, osc = converge(pools, world)
    regen = {}
    regen_path = OUT_DIR / "stage25_regen.jsonl"
    if regen_path.exists():
        for line in regen_path.open(encoding="utf-8"):
            r = json.loads(line)
            regen[r["iid"]] = r["answers"]
    final = {i: dict(s) for i, s in state.items()}
    for iid in contested_ids:
        final[iid]["answer"] = weighted_decide(
            pools[iid], world.context(final, iid), regen.get(iid, []), 0.5,
            iid in osc)
    base = metrics(pools, final, contested_ids)
    print(f"\n[0] BASELINE (shipped): {base}")
    results = {"baseline": base}

    # ---------- world-agreement per committed pool ----------
    lang_ans = defaultdict(dict)
    for iid, s in final.items():
        if s["answer"] is not None:
            lang_ans[iid.rsplit("_", 1)[-1]][pools[iid]["lang"]] = s["answer"]
    def world_share(iid, ans):
        prob = iid.rsplit("_", 1)[-1]
        lang = pools[iid]["lang"]
        others = [a for l, a in lang_ans[prob].items() if l != lang]
        return (sum(match(a, ans) for a in others) / len(others)) if others else 0.0

    # ---------- ARM A: harvest cleaning ----------
    harvest = []
    for (iid, k), r in scores.items():
        if not r["harvest"]:
            continue
        ans = final[iid]["answer"]
        if ans is None or not match(r["answer"], ans):
            continue
        harvest.append({"iid": iid, "lang": r["lang"], "p": r["p_clean"],
                        "genuine": match(ans, pools[iid]["gold"]),
                        "world": world_share(iid, ans)})
    gen = [h for h in harvest if h["genuine"]]
    con = [h for h in harvest if not h["genuine"]]
    print(f"\n[A] HARVEST: {len(harvest)} chains, "
          f"{100*len(con)/max(len(harvest),1):.2f}% contaminated")
    a_all = auc([h["p"] for h in gen], [h["p"] for h in con])
    a_world = auc([h["world"] for h in gen], [h["world"] for h in con])
    print(f"    p_clean AUC {a_all:.4f}   world-agreement AUC {a_world:.4f}")
    results["A_auc"] = {"p_clean": round(a_all, 4), "world": round(a_world, 4),
                        "n_genuine": len(gen), "n_contaminated": len(con)}

    print(f"    residue slices (world >= t: the vote is fooled there):")
    for t in (0.5, 0.7, 0.9):
        sub = [h for h in harvest if h["world"] >= t]
        g = [h["p"] for h in sub if h["genuine"]]
        c = [h["p"] for h in sub if not h["genuine"]]
        a = auc(g, c) if g and c else None
        print(f"      world>={t}: {len(g)} genuine / {len(c)} contaminated  "
              f"AUC {a if a else float('nan'):.4f}")
        results[f"A_residue_{t}"] = {"n_genuine": len(g), "n_contaminated": len(c),
                                     "auc": round(a, 4) if a else None}

    print(f"\n    two-stage frontier (world>=0.3 kept, then drop lowest p_clean):")
    kept = sorted([h for h in harvest if h["world"] >= 0.3], key=lambda h: h["p"])
    print(f"      {'drop':>6}{'kept':>9}{'contam':>8}{'purity':>9}{'genuine lost':>14}")
    frontier = []
    for drop in (0.0, 0.02, 0.05, 0.10):
        k = int(drop * len(kept))
        surv = kept[k:]
        c = sum(1 for h in surv if not h["genuine"])
        lost = sum(1 for h in kept[:k] if h["genuine"])
        purity = 100 * (1 - c / max(len(surv), 1))
        frontier.append({"drop": drop, "kept": len(surv), "contaminated": c,
                         "purity": round(purity, 3), "genuine_lost": lost})
        print(f"      {100*drop:>5.0f}%{len(surv):>9}{c:>8}{purity:>8.3f}%{lost:>14}")
    results["A_frontier"] = frontier

    # ---------- ARM B: in-loop tie-break ----------
    by_pool = defaultdict(dict)
    for (iid, k), r in scores.items():
        if r["contested"]:
            by_pool[iid].setdefault(r["answer"], []).append(r["p_clean"])
    changed = b_better = b_worse = 0
    state_b = {i: dict(s) for i, s in final.items()}
    for iid in contested_ids:
        cands = by_pool.get(iid)
        if not cands or len(cands) < 2:
            continue
        best = max(cands, key=lambda a: (max(cands[a]), a))
        cur = final[iid]["answer"]
        if cur is None or match(best, cur):
            continue
        changed += 1
        was_ok = match(cur, pools[iid]["gold"])
        now_ok = match(best, pools[iid]["gold"])
        if now_ok and not was_ok:
            b_better += 1
        elif was_ok and not now_ok:
            b_worse += 1
        state_b[iid]["answer"] = best
    mb = metrics(pools, state_b, contested_ids)
    print(f"\n[B] IN-LOOP TIE-BREAK: {changed} pools changed "
          f"({b_better} fixed, {b_worse} broken)")
    print(f"    {mb}")
    results["B"] = dict(mb, changed=changed, fixed=b_better, broken=b_worse)

    # ---------- ARM C: uPRM alone ----------
    state_c = {}
    for iid, pool in pools.items():
        cands = defaultdict(list)
        for k, chain in enumerate(pool["chains"]):
            r = scores.get((iid, k))
            if r:
                cands[chain["answer"]].append(r["p_clean"])
        if not cands:
            state_c[iid] = {"answer": None}
            continue
        state_c[iid] = {"answer": max(cands, key=lambda a: (max(cands[a]), a))}
    committed = [i for i, s in state_c.items() if s["answer"] is not None]
    hits = sum(match(state_c[i]["answer"], pools[i]["gold"]) for i in committed)
    print(f"\n[C] uPRM ALONE (no voting): coverage "
          f"{len(committed)/len(pools):.4f}  precision "
          f"{hits/max(len(committed),1):.4f}")
    print(f"    baseline for comparison: coverage {base['coverage']:.4f}  "
          f"precision {base['precision']:.4f}")
    results["C"] = {"coverage": round(len(committed)/len(pools), 4),
                    "precision": round(hits/max(len(committed), 1), 4)}

    (OUT_DIR / "stage36_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote stage36_results.json")
    print("DONE")


if __name__ == "__main__":
    main()
