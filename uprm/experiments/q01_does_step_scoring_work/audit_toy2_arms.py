import json, statistics as st
from collections import defaultdict
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc

rows=[json.loads(l) for l in open("stage33_chains.jsonl",encoding="utf-8")]
ARMS=("independent","joint","shuffled","poisoned")

# reconstruct n_steps by re-running stage33's build_sample deterministically
import random
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
from .toy1_first_broken_step import CHAIN_MAX_CHARS, MIN_STEPS, split_steps, first_bad_step
from . import toy2_joint_scoring as s33
pools=load_pools(s33.DEFAULT_ROOT)
rng=random.Random(s33.SEED)
sample=s33.build_sample(pools,rng)
print(f"rebuilt sample: {len(sample)} (jsonl has {len(rows)})")
ok = len(sample)==len(rows) and all(sample[i]["iid"]==rows[i]["iid"] and
        sample[i]["is_gold"]==rows[i]["is_gold"] for i in range(len(rows)))
print("order matches jsonl:", ok)
for i,r in enumerate(rows):
    r["n_steps"]=len(sample[i]["steps"])

def lm(sub, key):
    strata=defaultdict(list)
    for r in sub: strata[r["n_steps"]].append(r)
    p=c=0
    for s in strata.values():
        g=[r[key] for r in s if r["is_gold"]]
        b=[r[key] for r in s if not r["is_gold"]]
        if g and b: p+=len(g)*len(b); c+=auc(g,b)*len(g)*len(b)
    return c/p if p else float('nan'), p

print("\n=== AS REPORTED (each arm on its own surviving subset) ===")
print(f"{'arm':<14}{'n':>6}{'AUC':>9}{'len-matched':>13}{'pairs':>9}")
for a in ARMS:
    key=f"p_clean_{a}"
    sub=[r for r in rows if r.get(key) is not None]
    g=[r[key] for r in sub if r["is_gold"]]; b=[r[key] for r in sub if not r["is_gold"]]
    L,p=lm(sub,key)
    print(f"{a:<14}{len(sub):>6}{auc(g,b):>9.4f}{L:>13.4f}{p:>9}")

common=[r for r in rows if all(r.get(f"p_clean_{a}") is not None for a in ARMS)]
print(f"\n=== COMMON SUBSET, all four arms on the SAME {len(common)} chains ===")
print(f"{'arm':<14}{'n':>6}{'AUC':>9}{'len-matched':>13}{'pairs':>9}")
res={}
for a in ARMS:
    key=f"p_clean_{a}"
    g=[r[key] for r in common if r["is_gold"]]; b=[r[key] for r in common if not r["is_gold"]]
    A=auc(g,b); L,p=lm(common,key)
    res[a]=(A,L)
    print(f"{a:<14}{len(common):>6}{A:>9.4f}{L:>13.4f}{p:>9}")
print(f"\n  claim 'joint 0.848 vs independent 0.847 length-matched':")
print(f"    reported  joint 0.8481  independent 0.8469   diff +0.0012")
print(f"    matched   joint {res['joint'][1]:.4f}  independent {res['independent'][1]:.4f}   "
      f"diff {res['joint'][1]-res['independent'][1]:+.4f}")
print(f"  claim 'poisoned 0.724':  reported 0.7244   matched {res['poisoned'][1]:.4f}")

print("\n=== dropout is length-correlated ===")
for a in ARMS:
    inn=[r["n_steps"] for r in rows if r.get(f"p_clean_{a}") is not None]
    out=[r["n_steps"] for r in rows if r.get(f"p_clean_{a}") is None]
    print(f"  {a:<12} kept {len(inn):>4} mean steps {st.mean(inn):.2f}   "
          f"dropped {len(out):>4} mean steps {(st.mean(out) if out else float('nan')):.2f}")
print("\n=== class balance drift ===")
for a in ARMS:
    sub=[r for r in rows if r.get(f"p_clean_{a}") is not None]
    ng=sum(r["is_gold"] for r in sub)
    print(f"  {a:<12} {ng} gold / {len(sub)-ng} broken  (design was 350/350)")
