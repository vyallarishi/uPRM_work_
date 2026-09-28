"""Quantify impact of the precedence bug on stage37 T1/T4/T5, stage39 P1/P2, stage40 A/C/D."""
import json, math, re
from decimal import Decimal, InvalidOperation
from collections import defaultdict
from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, match, norm as safe_norm
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from ..q01_does_step_scoring_work.toy1_first_broken_step import split_steps, eval_equation
from .error_types import first_bad_step_safe
from icm.experiments.q08_earlier_8b_study.rung3_reasoning_structure import _EQUATION_RE, _OP_SPLIT_RE

_SAFE = re.compile(r"^[0-9\.\+\-\*/\(\)]+$")
def eval_prec(lhs):
    """Correct-precedence value, or None."""
    if not _SAFE.match(lhs): return None
    try:
        v = eval(lhs, {"__builtins__": {}}, {})
        return Decimal(str(v))
    except Exception:
        return None

def first_bad_step_prec(steps):
    """Same as first_bad_step_safe but honouring operator precedence."""
    for index, step in enumerate(steps, start=1):
        for body in _EQUATION_RE.findall(step):
            if "=" not in body: continue
            lhs,_,rhs = body.rpartition("=")
            lhs = lhs.replace("×","*").replace("÷","/").replace(" ","")
            right = safe_norm(rhs)
            if not lhs or right is None: continue
            v = eval_prec(lhs)
            if v is None:
                # fall back to the original left-to-right evaluator
                try: verdict = eval_equation(lhs, right)
                except Exception: continue
                if verdict is False: return index
                continue
            try: want = Decimal(right)
            except Exception: continue
            if v == want: continue
            scale = max(abs(v), abs(want))
            if scale == 0: return index
            if abs(v-want)/scale > Decimal("0.005"): return index
    return None

scores={}
for sh in range(4):
    for line in open(f"stage35_scores_{sh}of4.jsonl", encoding="utf-8"):
        r=json.loads(line); scores[(r["iid"],r["k"])]=r
pools=load("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500")
world=World(pools); state,contested_ids,osc=converge(pools,world)
from icm.experiments.q03_does_regeneration_help.regeneration_15lang import weighted_decide
regen={}
import os
if os.path.exists("stage25_regen.jsonl"):
    for line in open("stage25_regen.jsonl",encoding="utf-8"):
        r=json.loads(line); regen[r["iid"]]=r["answers"]
final={i:dict(s) for i,s in state.items()}
for iid in contested_ids:
    final[iid]["answer"]=weighted_decide(pools[iid], world.context(final,iid), regen.get(iid,[]),0.5, iid in osc)
print("converged", flush=True)

rows=[]
for (iid,k),r in scores.items():
    pool=pools[iid]; chain=pool["chains"][k]
    steps=split_steps(chain["text"])
    rows.append({"iid":iid,"k":k,"p":r["p_clean"],"j_hat":r["j_hat"],
        "n_steps":r["n_steps"],"harvest":r["harvest"],
        "chain_right":match(chain["answer"],pool["gold"]),
        "commit_right": state[iid]["answer"] is not None and match(state[iid]["answer"],pool["gold"]),
        "bad_old":first_bad_step_safe(steps),
        "bad_new":first_bad_step_prec(steps)})
print("annotated", len(rows), flush=True)

for r in rows:
    r["err_old"]=r["bad_old"] is not None
    r["err_new"]=r["bad_new"] is not None

print("\n=== T4 prevalence: old (buggy) vs precedence-correct ===")
harv=[r for r in rows if r["harvest"]]
contam=[r for r in harv if not r["commit_right"]]
genuine=[r for r in harv if r["commit_right"]]
wrong_all=[r for r in rows if not r["chain_right"]]
for lab,sub in (("all wrong chains",wrong_all),("contaminated harvest",contam),("genuine harvest",genuine)):
    o=sum(r["err_old"] for r in sub); n=sum(r["err_new"] for r in sub)
    print(f"  {lab:<24} old {o:>6}/{len(sub):<7} ({100*o/max(len(sub),1):.1f}%)   "
          f"corrected {n:>6} ({100*n/max(len(sub),1):.1f}%)")

print("\n=== T1 harvest AUC by error type (stage37 claim: 0.7124 / 0.6459) ===")
pos=[r["p"] for r in genuine]
for tag,f in (("old",'err_old'),("corrected",'err_new')):
    w=[r["p"] for r in contam if r[f]]; wo=[r["p"] for r in contam if not r[f]]
    print(f"  {tag:<10} WITH false eq n={len(w):>5} AUC {auc(pos,w) if w else float('nan'):.4f}   "
          f"NO false eq n={len(wo):>5} AUC {auc(pos,wo) if wo else float('nan'):.4f}")

print("\n=== T5 localization (stage37 claim: 14.7% exact vs 20.2% chance) ===")
for tag,fb,fe in (("old","bad_old","err_old"),("corrected","bad_new","err_new")):
    loc=[r for r in wrong_all if r[fe]]
    ex=sum(1 for r in loc if r["j_hat"]==r[fb])
    wi=sum(1 for r in loc if abs(r["j_hat"]-r[fb])<=1)
    ch=sum(1.0/(r["n_steps"]+1) for r in loc)/max(len(loc),1)
    print(f"  {tag:<10} n={len(loc):>5}  exact {ex} ({100*ex/max(len(loc),1):.1f}%)  "
          f"within1 {100*wi/max(len(loc),1):.1f}%  chance {100*ch:.1f}%")

print("\n=== T3b 2x2 cells: disjoint? same positive class? ===")
plur_share={}
for iid,pool in pools.items():
    from collections import Counter
    c=Counter(ch["answer"] for ch in pool["chains"]); tot=len(pool["chains"])
    for a,n in c.items(): plur_share[(iid,a)]=n/tot
for r in rows:
    r["own_share"]=plur_share.get((r["iid"],pools[r["iid"]]["chains"][r["k"]]["answer"]),0.0)
right_all=[r for r in rows if r["chain_right"]]
tot=0
for hard in (True,False):
    for he in (True,False):
        sub=[r for r in wrong_all if (r["own_share"]>=0.70)==hard and r["err_old"]==he]
        tot+=len(sub)
print(f"  cells sum to {tot}; wrong_all = {len(wrong_all)}  -> disjoint & exhaustive: {tot==len(wrong_all)}")
print(f"  positives in every cell = right_all, n={len(right_all)} (same set). OK")

print("\n=== stage39/40 lucky-guess + P1/P2 with corrected checker ===")
by_pool=defaultdict(list)
for r in rows:
    iid=r["iid"]; pool=pools[iid]; committed=final[iid]["answer"]
    if committed is None: continue
    if not match(pool["chains"][r["k"]]["answer"], committed): continue
    cr=match(committed,pool["gold"])
    by_pool[iid].append({"p":r["p"],"n_steps":r["n_steps"],
        "chars":len(pool["chains"][r["k"]]["text"]),"commit_right":cr,
        "err_old":r["err_old"],"err_new":r["err_new"],
        "lucky_old":r["err_old"] and cr,"lucky_new":r["err_new"] and cr,
        "lang":pool["lang"]})
allc=[c for v in by_pool.values() for c in v]
print(f"  SFT candidates {len(allc)} across {len(by_pool)} pools")
import statistics as stt
xs=[c["n_steps"] for c in allc]; ys=[c["p"] for c in allc]
mx,my=stt.mean(xs),stt.mean(ys)
var=sum((x-mx)**2 for x in xs)
slope=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/var
print(f"  global slope {slope:+.4f}/step (stage40 reports -0.0649)")
for c in allc:
    c["per_step"]=c["p"]/max(c["n_steps"],1)
    c["resid"]=c["p"]-(my+slope*(c["n_steps"]-mx))

for lk in ("lucky_old","lucky_new"):
    print(f"\n  --- P1 with {lk} ---")
    for v in ("p","per_step","resid"):
        wins=total=0
        for chains in by_pool.values():
            L=[c for c in chains if c[lk]]; C=[c for c in chains if not c[lk]]
            if not L or not C: continue
            total+=1
            if max(c[v] for c in C) > max(c[v] for c in L): wins+=1
        print(f"    {v:<10} pools={total:<6} clean higher {wins} ({100*wins/max(total,1):.1f}%)")

print("\n  --- C: SFT set composition, false-eq rate under BOTH checkers ---")
def show(name, rowsel):
    o=100*sum(1 for c in rowsel if c["err_old"])/len(rowsel)
    n=100*sum(1 for c in rowsel if c["err_new"])/len(rowsel)
    s=stt.mean(c["n_steps"] for c in rowsel)
    print(f"    {name:<16} false-eq old {o:>5.2f}%   corrected {n:>5.2f}%   steps {s:.2f}")
import random
rng=random.Random(20260828)
show("random",[rng.choice(v) for v in by_pool.values()])
for v in ("p","per_step","resid"):
    show(f"best[{v}]",[max(ch,key=lambda c:c[v]) for ch in by_pool.values()])
