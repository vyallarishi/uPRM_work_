import json, os
from collections import defaultdict
from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import auc, match
from icm.experiments.q03_does_regeneration_help.regeneration_15lang import weighted_decide
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import World, converge, load
from icm.experiments.q01_can_voting_label_data.consensus_loop_15lang import metrics

scores={}
for sh in range(4):
    for line in open(f"stage35_scores_{sh}of4.jsonl",encoding="utf-8"):
        r=json.loads(line); scores[(r["iid"],r["k"])]=r
pools=load("/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500")
world=World(pools); state,contested_ids,osc=converge(pools,world)
regen={}
if os.path.exists("stage25_regen.jsonl"):
    for line in open("stage25_regen.jsonl",encoding="utf-8"):
        r=json.loads(line); regen[r["iid"]]=r["answers"]
final={i:dict(s) for i,s in state.items()}
for iid in contested_ids:
    final[iid]["answer"]=weighted_decide(pools[iid],world.context(final,iid),regen.get(iid,[]),0.5,iid in osc)
base=metrics(pools,final,contested_ids)
print("baseline:",base)

by_pool=defaultdict(dict)
for (iid,k),r in scores.items():
    if r["contested"]:
        by_pool[iid].setdefault(r["answer"],[]).append(r["p_clean"])
changed=b_better=b_worse=b_neutral=0
state_b={i:dict(s) for i,s in final.items()}
skipped_nocand=skipped_1cand=0
for iid in contested_ids:
    cands=by_pool.get(iid)
    if not cands: skipped_nocand+=1; continue
    if len(cands)<2: skipped_1cand+=1; continue
    best=max(cands,key=lambda a:(max(cands[a]),a))
    cur=final[iid]["answer"]
    if cur is None or match(best,cur): continue
    changed+=1
    was=match(cur,pools[iid]["gold"]); now=match(best,pools[iid]["gold"])
    if now and not was: b_better+=1
    elif was and not now: b_worse+=1
    else: b_neutral+=1
    state_b[iid]["answer"]=best
print(f"Arm B: changed {changed} = fixed {b_better} + broken {b_worse} + neutral {b_neutral}"
      f"  (sum {b_better+b_worse+b_neutral})")
print(f"  contested pools with NO scored cands: {skipped_nocand}; with only 1 candidate: {skipped_1cand}")
print("  reproduces stage36 log (431 fixed / 3967 broken):",
      b_better==431 and b_worse==3967)
print("Arm B metrics:", metrics(pools,state_b,contested_ids))

# ARM C fairness
state_c={}
for iid,pool in pools.items():
    cands=defaultdict(list)
    for k,ch in enumerate(pool["chains"]):
        r=scores.get((iid,k))
        if r: cands[ch["answer"]].append(r["p_clean"])
    state_c[iid]={"answer": (max(cands,key=lambda a:(max(cands[a]),a)) if cands else None)}
committed=[i for i,s in state_c.items() if s["answer"] is not None]
hits=sum(match(state_c[i]["answer"],pools[i]["gold"]) for i in committed)
print(f"\nArm C: coverage {len(committed)/len(pools):.4f} precision {hits/max(len(committed),1):.4f}")

# CRITICAL: what candidate set did Arm C actually see?
n_all=n_restricted=0
tot_ans_pool=0; tot_ans_scored=0
for iid,pool in pools.items():
    allans={ch["answer"] for ch in pool["chains"] if ch["answer"] is not None}
    scored={ch["answer"] for k,ch in enumerate(pool["chains"]) if (iid,k) in scores}
    tot_ans_pool+=len(allans); tot_ans_scored+=len(scored)
    if scored and scored<allans: n_restricted+=1
    elif scored: n_all+=1
print(f"  pools where the SCORED candidate set is a strict SUBSET of the pool's answers: {n_restricted}")
print(f"  pools where all answers were scored: {n_all}")
print(f"  mean distinct answers per pool: in pool {tot_ans_pool/len(pools):.2f}, scored {tot_ans_scored/len(pools):.2f}")
print("  -> stage35 only scored chains that are HARVEST (agree with the committed")
print("     answer) or in a CONTESTED pool. In a settled (non-contested) pool only")
print("     the committed answer's chains were scored, so Arm C has exactly ONE")
print("     candidate there and trivially 'picks' the pipeline's answer.")

# quantify: how many of Arm C's committed pools had >1 candidate to choose from?
multi=0; multi_hits=0; single=0; single_hits=0
for iid in committed:
    cands=defaultdict(list)
    for k,ch in enumerate(pools[iid]["chains"]):
        if (iid,k) in scores: cands[ch["answer"]].append(1)
    ok=match(state_c[iid]["answer"],pools[iid]["gold"])
    if len(cands)>1: multi+=1; multi_hits+=ok
    else: single+=1; single_hits+=ok
print(f"\n  Arm C committed pools with >1 scored candidate: {multi} (precision {multi_hits/max(multi,1):.4f})")
print(f"  Arm C committed pools with exactly 1 scored candidate: {single} (precision {single_hits/max(single,1):.4f})")
print(f"  -> the headline 86.89% is {100*single/len(committed):.1f}% inherited from the pipeline,")
print(f"     not chosen by the PRM.")
