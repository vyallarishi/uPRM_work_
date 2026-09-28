"""AUDIT: stage33 Toy2 arms compared on DIFFERENT chain subsets?"""
import json
from collections import defaultdict
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc

rows=[json.loads(l) for l in open("stage33_chains.jsonl",encoding="utf-8")]
ARMS=("independent","joint","shuffled","poisoned")
print(f"{len(rows)} chains total\n")
print(f"{'arm':<14}{'n scored':>10}{'n gold':>9}{'n broken':>10}")
sets={}
for a in ARMS:
    s=[r for r in rows if r.get(f"p_clean_{a}") is not None]
    sets[a]=set(id(r) for r in s)
    print(f"{a:<14}{len(s):>10}{sum(r['is_gold'] for r in s):>9}{sum(not r['is_gold'] for r in s):>10}")

common=set.intersection(*sets.values())
print(f"\nchains scored in ALL FOUR arms: {len(common)}")
print("-> the reported per-arm AUCs are computed on DIFFERENT populations.")
print("   Chains drop out when the prompt exceeds MAX_PROMPT_TOKENS=3800, and")
print("   the joint/poisoned arms add 7 neighbour chains, so LONG chains are")
print("   systematically dropped there but kept in 'independent'.")

print("\n=== recompute all arms on the COMMON subset (apples-to-apples) ===")
cm=[r for r in rows if id(r) in common]
print(f"{'arm':<14}{'n':>6}{'AUC':>9}{'len-matched':>13}")
res={}
for a in ARMS:
    key=f"p_clean_{a}"
    g=[r[key] for r in cm if r["is_gold"]]
    b=[r[key] for r in cm if not r["is_gold"]]
    A=auc(g,b)
    strata=defaultdict(list)
    for r in cm: strata[r["n_steps"]].append(r)
    p=c=0
    for sub in strata.values():
        gg=[r[key] for r in sub if r["is_gold"]]
        bb=[r[key] for r in sub if not r["is_gold"]]
        if gg and bb:
            p+=len(gg)*len(bb); c+=auc(gg,bb)*len(gg)*len(bb)
    lm=c/p if p else float('nan')
    res[a]=(A,lm)
    print(f"{a:<14}{len(cm):>6}{A:>9.4f}{lm:>13.4f}")
print(f"\nP1 joint-independent  reported {0.7906-0.8112:+.4f}  common-subset {res['joint'][0]-res['independent'][0]:+.4f}")
print(f"P2 joint-shuffled     reported {0.7906-0.7472:+.4f}  common-subset {res['joint'][0]-res['shuffled'][0]:+.4f}")
print(f"P3 joint-poisoned     reported {0.7906-0.7333:+.4f}  common-subset {res['joint'][0]-res['poisoned'][0]:+.4f}")
print(f"\nlength-matched joint vs independent: reported 0.8481 vs 0.8469 (+0.0012);"
      f" common-subset {res['joint'][1]:.4f} vs {res['independent'][1]:.4f} "
      f"({res['joint'][1]-res['independent'][1]:+.4f})")
print(f"poisoned length-matched: reported 0.7244; common-subset {res['poisoned'][1]:.4f}")

# n_steps distribution by arm-membership
import statistics as st
print("\n=== dropout is length-correlated ===")
for a in ("joint","shuffled","poisoned"):
    inn=[r["n_steps"] for r in rows if r.get(f"p_clean_{a}") is not None]
    out=[r["n_steps"] for r in rows if r.get(f"p_clean_{a}") is None]
    print(f"  {a:<12} kept n={len(inn)} mean steps {st.mean(inn):.2f}   "
          f"dropped n={len(out)} mean steps {st.mean(out) if out else float('nan'):.2f}")
