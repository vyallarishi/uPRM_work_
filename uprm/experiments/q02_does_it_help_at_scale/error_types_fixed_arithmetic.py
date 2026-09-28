"""Stage 37-R (RERUN, precedence-correct arithmetic via arith_fixed) — WHY did the PRM's AUC collapse from 0.891 to 0.646?

Two explanations fit the facts equally well so far, and they imply different
things about whether the method is fixable:

  H_ERRORTYPE  The PRM finds BROKEN COMPUTATIONS. Toy 1's broken chains
               included many of those; harvest contamination is dominated by
               COMPREHENSION errors (misread the question, skipped a
               constraint) where every individual step is locally valid and
               there is no step to point at. Prediction: within the harvest,
               AUC should be clearly higher on contaminated chains that
               contain a verifiable false equation than on those that don't.

  H_HARDNEG    Nothing about error type; harvest contamination is simply a
               HARDER negative class. Those chains won a landslide vote AND
               survived the world check, so they are the most plausible wrong
               chains that exist. Toy 1's negatives were any wrong chain.
               Prediction: AUC is flat across error types, but drops sharply
               when Toy-1-style negatives are restricted to landslide winners.

TESTS (all CPU, on the stage35 cache + the rung3 arithmetic checker):
  T1 harvest AUC split by error type (arithmetic-error vs none)
  T2 harvest AUC split by pool difficulty (did the wrong answer win big?)
  T3 the reverse: on ALL scored chains, AUC over wrong-chains that WOULD have
     won a landslide vs those that would not -- isolates hardness from type
  T4 prevalence: what fraction of contaminated chains contain a checkable
     computational error at all
  T5 localization on the harvest's arithmetic-error subset: when there IS a
     broken step, does j_hat find it? (Toy 1: 17.6% vs 21.7% chance)

READING THE RESULT:
  T1 large gap + T3 flat  -> H_ERRORTYPE. The method works on computational
     errors and our data has few; that is a statement about PRMs, and the fix
     would be a different error taxonomy, not a better prompt.
  T1 flat + T3 large gap  -> H_HARDNEG. Error type is a red herring; the PRM
     is reading plausibility, and harder negatives will always defeat it.
  Both large -> both effects are real; report both.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, match
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from ..q01_does_step_scoring_work.toy1_first_broken_step import split_steps
from icm.experiments.q08_earlier_8b_study.rung3_reasoning_structure import _EQUATION_RE
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import norm as safe_norm


from icm.experiments.q07_audits.fix_arithmetic_precedence import first_bad_step as first_bad_step_safe


ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"
from ...paths import RUN_DIR as OUT_DIR  # was: Path(__file__).parent
LANDSLIDE = 0.70


def main():
    scores = {}
    for shard in range(4):
        for line in (OUT_DIR / f"stage35_scores_{shard}of4.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            scores[(r["iid"], r["k"])] = r
    print(f"loaded {len(scores)} scored chains", flush=True)

    pools = load(ROOT)
    world = World(pools)
    state, contested_ids, _ = converge(pools, world)
    print("converged", flush=True)

    # Annotate every scored chain: does it contain a verifiable false equation?
    rows = []
    for (iid, k), r in scores.items():
        pool = pools[iid]
        chain = pool["chains"][k]
        steps = split_steps(chain["text"])
        bad = first_bad_step_safe(steps)
        committed = state[iid]["answer"]
        rows.append({
            "iid": iid, "k": k, "lang": r["lang"], "p": r["p_clean"],
            "j_hat": r["j_hat"], "n_steps": r["n_steps"],
            "answer": chain["answer"], "gold": pool["gold"],
            "chain_right": match(chain["answer"], pool["gold"]),
            "harvest": r["harvest"],
            "committed": committed,
            "commit_right": (committed is not None
                             and match(committed, pool["gold"])),
            "share": pool["share"],
            "bad_step": bad,
            "has_arith_error": bad is not None,
        })
    print(f"annotated {len(rows)} chains", flush=True)
    results = {}

    def A(sub, label, indent="  "):
        g = [r["p"] for r in sub if r["chain_right"]]
        b = [r["p"] for r in sub if not r["chain_right"]]
        a = auc(g, b) if g and b else None
        print(f"{indent}{label:<46} {len(g):>7} right / {len(b):>6} wrong"
              f"   AUC {a if a is not None else float('nan'):.4f}")
        return {"n_right": len(g), "n_wrong": len(b),
                "auc": round(a, 4) if a is not None else None}

    # ---------------- T4 prevalence ----------------
    print("\n=== T4: how often is there a checkable computational error? ===")
    harvest = [r for r in rows if r["harvest"]]
    contam = [r for r in harvest if not r["commit_right"]]
    genuine = [r for r in harvest if r["commit_right"]]
    wrong_all = [r for r in rows if not r["chain_right"]]
    for label, sub in (("all wrong chains", wrong_all),
                       ("contaminated harvest chains", contam),
                       ("genuine harvest chains", genuine)):
        n_bad = sum(r["has_arith_error"] for r in sub)
        print(f"  {label:<34} {n_bad:>7}/{len(sub):<7} "
              f"({100*n_bad/max(len(sub),1):.1f}% contain a false equation)")
        results.setdefault("T4", {})[label] = {
            "n": len(sub), "with_arith_error": n_bad,
            "pct": round(100*n_bad/max(len(sub), 1), 2)}

    # ---------------- T1 harvest AUC by error type ----------------
    # genuine chains are the positives; split the NEGATIVES by error type
    print("\n=== T1: harvest AUC, contaminated split by error type ===")
    print("    (positives = genuine harvest chains, held fixed)")
    pos = [r["p"] for r in genuine]
    for label, sub in (("contaminated WITH a false equation",
                        [r for r in contam if r["has_arith_error"]]),
                       ("contaminated with NO false equation",
                        [r for r in contam if not r["has_arith_error"]])):
        neg = [r["p"] for r in sub]
        a = auc(pos, neg) if neg else None
        print(f"  {label:<46} {len(neg):>6} chains   "
              f"AUC {a if a is not None else float('nan'):.4f}")
        results.setdefault("T1", {})[label] = {
            "n_contaminated": len(neg), "auc": round(a, 4) if a else None}

    # ---------------- T2 harvest AUC by how big the wrong answer won -------
    print("\n=== T2: harvest AUC by how decisively the wrong answer won ===")
    for lo, hi in ((0.0, 0.75), (0.75, 0.9), (0.9, 1.01)):
        sub = [r for r in contam if lo <= r["share"] < hi]
        neg = [r["p"] for r in sub]
        a = auc(pos, neg) if neg else None
        print(f"  local share {lo:.2f}-{hi:.2f}: {len(neg):>6} contaminated   "
              f"AUC {a if a is not None else float('nan'):.4f}")
        results.setdefault("T2", {})[f"share_{lo}_{hi}"] = {
            "n": len(neg), "auc": round(a, 4) if a else None}

    # ---------------- T3 hardness, isolated from error type ----------------
    # Toy-1-style task (any right vs any wrong chain), but split the wrong
    # chains by whether their answer won a landslide in its own pool.
    print("\n=== T3: Toy-1-style AUC, wrong chains split by pool hardness ===")
    plur_share = {}
    for iid, pool in pools.items():
        counts = Counter(c["answer"] for c in pool["chains"])
        total = len(pool["chains"])
        for ans, n in counts.items():
            plur_share[(iid, ans)] = n / total
    for r in rows:
        r["own_share"] = plur_share.get((r["iid"], r["answer"]), 0.0)
    right_all = [r for r in rows if r["chain_right"]]
    print(f"  {'wrong chains whose answer...':<46} {'n':>7}          AUC")
    for label, sub in (("won a landslide (>=0.70) in its pool",
                        [r for r in wrong_all if r["own_share"] >= LANDSLIDE]),
                       ("did NOT win a landslide",
                        [r for r in wrong_all if r["own_share"] < LANDSLIDE])):
        neg = [r["p"] for r in sub]
        a = auc([r["p"] for r in right_all], neg) if neg else None
        print(f"  {label:<46} {len(neg):>7}   AUC "
              f"{a if a is not None else float('nan'):.4f}")
        results.setdefault("T3", {})[label] = {
            "n_wrong": len(neg), "auc": round(a, 4) if a else None}

    # cross-tab: hardness x error type, the clean 2x2
    print("\n=== T3b: 2x2 -- hardness x error type (wrong chains) ===")
    print(f"  {'':<26}{'has false eq':>16}{'no false eq':>16}")
    grid = {}
    for hard_label, hard in (("landslide winner", True), ("not a winner", False)):
        line = f"  {hard_label:<26}"
        for has_err in (True, False):
            sub = [r for r in wrong_all
                   if (r["own_share"] >= LANDSLIDE) == hard
                   and r["has_arith_error"] == has_err]
            a = auc([r["p"] for r in right_all], [r["p"] for r in sub]) if sub else None
            grid[f"{hard_label}|{'err' if has_err else 'noerr'}"] = {
                "n": len(sub), "auc": round(a, 4) if a else None}
            line += f"{(f'{a:.3f}' if a else 'n/a') + f' (n={len(sub)})':>16}"
        print(line)
    results["T3b"] = grid

    # ---------------- T5 localization where a real error exists ------------
    print("\n=== T5: localization on chains WITH a verifiable broken step ===")
    loc = [r for r in wrong_all if r["has_arith_error"]]
    exact = sum(1 for r in loc if r["j_hat"] == r["bad_step"])
    within = sum(1 for r in loc if abs(r["j_hat"] - r["bad_step"]) <= 1)
    chance = sum(1.0 / (r["n_steps"] + 1) for r in loc) / max(len(loc), 1)
    missed = sum(1 for r in loc if r["j_hat"] == r["n_steps"] + 1)
    print(f"  n = {len(loc)}")
    print(f"  exact hit          {exact:>6} ({100*exact/max(len(loc),1):.1f}%)"
          f"   chance {100*chance:.1f}%")
    print(f"  within one step    {within:>6} ({100*within/max(len(loc),1):.1f}%)")
    print(f"  said 'no error'    {missed:>6} ({100*missed/max(len(loc),1):.1f}%)")
    results["T5"] = {"n": len(loc), "exact": exact, "within_one": within,
                     "said_clean": missed,
                     "exact_rate": round(exact/max(len(loc), 1), 4),
                     "chance_rate": round(chance, 4)}

    (OUT_DIR / "stage37R_results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote stage37R_results.json")
    print("DONE")


if __name__ == "__main__":
    main()
