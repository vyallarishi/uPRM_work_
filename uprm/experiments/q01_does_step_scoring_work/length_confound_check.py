"""Is stage32's 0.835 real, or a length artifact? (CPU, cached scores)

p_clean = log P(j = T+1) = sum of T per-step log P(+). More steps -> more
negative terms -> mechanically lower score. If broken chains are longer,
0.835 could be pure length. Checks:
  1. AUC of NEGATIVE step count alone (zero-model baseline)
  2. AUC of p_clean after regressing out step count
  3. AUC of the length-NORMALIZED score (p_clean / n_steps)
  4. Length-matched AUC: within each step-count stratum, then pooled
  5. Per-language and per-tier breakdown
  6. Distribution of n_steps by class
"""
import json, statistics as st
from collections import defaultdict
from icm.experiments.q02_can_the_model_judge_itself import verdict_judge_gate as gate
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc
from ...paths import RUN_DIR

rows = [json.loads(l) for l in
        open(f"{RUN_DIR}/stage32_chains.jsonl", encoding="utf-8")]
for r in rows:
    r["lang"] = r["iid"].split("_")[1]
print(f"{len(rows)} chains "
      f"({sum(r['is_gold'] for r in rows)} gold / "
      f"{sum(not r['is_gold'] for r in rows)} non-gold)")

def A(vals_fn, subset=None):
    sub = subset if subset is not None else rows
    g = [vals_fn(r) for r in sub if r["is_gold"]]
    b = [vals_fn(r) for r in sub if not r["is_gold"]]
    return auc(g, b) if g and b else None

print("\n=== 6. step counts by class ===")
gl = [r["n_steps"] for r in rows if r["is_gold"]]
bl = [r["n_steps"] for r in rows if not r["is_gold"]]
print(f"  clean : mean {st.mean(gl):.2f}  median {st.median(gl):.0f}")
print(f"  broken: mean {st.mean(bl):.2f}  median {st.median(bl):.0f}")

print("\n=== 1. zero-model baseline: shorter = cleaner ===")
a_len = A(lambda r: -r["n_steps"])
print(f"  AUC(-n_steps alone) = {a_len:.4f}")
print(f"  AUC(p_clean)        = {A(lambda r: r['p_clean']):.4f}")

print("\n=== 2. p_clean with step count regressed out ===")
xs = [r["n_steps"] for r in rows]
ys = [r["p_clean"] for r in rows]
mx, my = st.mean(xs), st.mean(ys)
var = sum((x-mx)**2 for x in xs)
slope = sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / var if var else 0.0
for r in rows:
    r["resid"] = r["p_clean"] - (my + slope*(r["n_steps"]-mx))
ss_tot = sum((y-my)**2 for y in ys)
ss_res = sum(r["resid"]**2 for r in rows)
print(f"  slope {slope:+.4f} per step, R^2 = {1-ss_res/ss_tot:.4f}")
print(f"  AUC(residual) = {A(lambda r: r['resid']):.4f}")

print("\n=== 3. length-normalized score ===")
print(f"  AUC(p_clean / n_steps) = {A(lambda r: r['p_clean']/max(r['n_steps'],1)):.4f}")

print("\n=== 4. length-matched (within step-count strata) ===")
strata = defaultdict(list)
for r in rows: strata[r["n_steps"]].append(r)
tot_pairs = tot_conc = 0
print(f"  {'steps':>6}{'clean':>7}{'broken':>8}{'AUC':>8}")
for n in sorted(strata):
    sub = strata[n]
    g = [r["p_clean"] for r in sub if r["is_gold"]]
    b = [r["p_clean"] for r in sub if not r["is_gold"]]
    if not g or not b: continue
    a = auc(g, b)
    print(f"  {n:>6}{len(g):>7}{len(b):>8}{a:>8.4f}")
    tot_pairs += len(g)*len(b); tot_conc += a*len(g)*len(b)
print(f"  POOLED length-matched AUC = {tot_conc/max(tot_pairs,1):.4f} "
      f"({tot_pairs} pairs)")

print("\n=== 5. per-tier and per-language ===")
for tier in ("high","mid","low"):
    sub = [r for r in rows if gate.TIER.get(r["lang"]) == tier]
    a = A(lambda r: r["p_clean"], sub)
    ng = sum(r["is_gold"] for r in sub)
    print(f"  {tier:<5} n={len(sub):>4} ({ng} clean) AUC "
          f"{'n/a' if a is None else f'{a:.4f}'}")
print()
for lang in sorted({r["lang"] for r in rows}):
    sub = [r for r in rows if r["lang"] == lang]
    a = A(lambda r: r["p_clean"], sub)
    ng = sum(r["is_gold"] for r in sub)
    if a is not None:
        print(f"  {lang:<4} n={len(sub):>3} ({ng} clean) AUC {a:.4f}")
print("DONE")
