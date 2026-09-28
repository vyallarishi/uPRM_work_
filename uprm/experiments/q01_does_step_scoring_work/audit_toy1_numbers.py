"""AUDIT: Toy1 0.835/0.891/20782 pairs, localization; Toy2 numbers; length-matching method."""
import json, statistics as st
from collections import defaultdict
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc
from icm.experiments.q02_can_the_model_judge_itself import verdict_judge_gate as gate

rows=[json.loads(l) for l in open("stage32_chains.jsonl",encoding="utf-8")]
for r in rows: r["lang"]=r["iid"].split("_")[1]
print(f"stage32: {len(rows)} chains ({sum(r['is_gold'] for r in rows)} gold / "
      f"{sum(not r['is_gold'] for r in rows)} non-gold)")
g=[r["p_clean"] for r in rows if r["is_gold"]]
b=[r["p_clean"] for r in rows if not r["is_gold"]]
print(f"  raw AUC = {auc(g,b):.4f}  [claim 0.835]")
print(f"  AUC(-n_steps) = {auc([-r['n_steps'] for r in rows if r['is_gold']],[-r['n_steps'] for r in rows if not r['is_gold']]):.4f} [claim 0.448]")

strata=defaultdict(list)
for r in rows: strata[r["n_steps"]].append(r)
pairs=conc=0
print(f"  {'steps':>6}{'clean':>7}{'broken':>8}{'AUC':>9}{'pairs':>9}")
for n in sorted(strata):
    sub=strata[n]
    gg=[r["p_clean"] for r in sub if r["is_gold"]]
    bb=[r["p_clean"] for r in sub if not r["is_gold"]]
    if not gg or not bb: continue
    a=auc(gg,bb); p=len(gg)*len(bb)
    print(f"  {n:>6}{len(gg):>7}{len(bb):>8}{a:>9.4f}{p:>9}")
    pairs+=p; conc+=a*p
print(f"  POOLED length-matched AUC = {conc/pairs:.4f} over {pairs} pairs  [claim 0.891 / 20,782]")

print("\n  --- localization (claim 17.6% vs 21.7% chance) ---")
loc=[r for r in rows if not r["is_gold"] and r["bad_step"]]
hits=sum(1 for r in loc if r["j_hat"]==r["bad_step"])
chance=sum(1.0/(r["n_steps"]+1) for r in loc)/len(loc)
print(f"  n={len(loc)}  exact {hits} ({100*hits/len(loc):.1f}%)  chance {100*chance:.1f}%")
print(f"  NOTE: chance uses 1/(T+1) = uniform over ALL T+1 hypotheses, but the")
print(f"  hit event only has T possible successes (j_hat==bad_step, bad_step<=T).")
print(f"  A uniform-random j over 1..T+1 hits with prob 1/(T+1): that IS correct.")

print("\n=== stage33 Toy 2 ===")
r33=[json.loads(l) for l in open("stage33_chains.jsonl",encoding="utf-8")]
print(f"  {len(r33)} chains")
res33=json.load(open("stage33_results.json"))
print(json.dumps(res33,indent=2))

print("\n=== length-matching methodology check ===")
print("  Pooled = sum_k AUC_k * n_k / sum_k n_k where n_k = |clean_k|*|broken_k|.")
print("  This is a weighted mean of per-stratum AUCs = (total concordant pairs)/(total pairs).")
print("  It is a legitimate stratified estimator, BUT: strata with only one class")
print("  contribute ZERO pairs and are silently DROPPED. Count how many:")
dropped=sum(1 for n in strata if not([r for r in strata[n] if r["is_gold"]] and [r for r in strata[n] if not r["is_gold"]]))
n_drop=sum(len(strata[n]) for n in strata if not([r for r in strata[n] if r["is_gold"]] and [r for r in strata[n] if not r["is_gold"]]))
print(f"  stage32: {dropped} strata dropped, {n_drop} chains excluded out of {len(rows)}")
