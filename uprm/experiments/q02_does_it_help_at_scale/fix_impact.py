"""What do the two fixes actually change on real data?

Measures, without re-running any GPU work:
  1. arith_fixed vs the old eval_equation on the stage35 population
     (the false-positive rate the audit reported, verified independently)
  2. diverse8_fixed vs stage17's version on the full37-1500 pools:
     landslide/contested composition, plurality changes, and the resulting
     converged coverage/precision -- i.e. how much the headline moves.
"""
import json
from collections import Counter, defaultdict

from icm.experiments.q01_can_voting_label_data import consensus_loop_37lang as s17
from icm.experiments.q02_can_the_model_judge_itself.verdict_judge_gate import load_pools, match
from icm.experiments.q01_can_voting_label_data.consensus_loop_15lang import metrics
from icm.experiments.q04_full_scale_headline.full_scale_37x1500 import World, converge
from icm.experiments.q07_audits import fix_arithmetic_precedence as arith_fixed
from icm.experiments.q07_audits import fix_pool_view_share as diverse8_fixed
from ..q01_does_step_scoring_work.toy1_first_broken_step import split_steps
from .error_types import first_bad_step_safe as old_first_bad

ROOT = "/scratch/rvyalla/multilingual-icm/results/qwen3-14b-full37-1500"

print("=" * 68)
print("FIX 1 — operator precedence in the arithmetic checker")
print("=" * 68)
raw = load_pools(ROOT)
old_flagged = new_flagged = both = only_old = only_new = 0
examples = []
for iid in sorted(raw):
    for chain in raw[iid]["chains"]:
        steps = split_steps(chain["text"])
        if len(steps) < 2:
            continue
        o = old_first_bad(steps)
        n = arith_fixed.first_bad_step(steps)
        old_flagged += o is not None
        new_flagged += n is not None
        if o is not None and n is not None:
            both += 1
        elif o is not None:
            only_old += 1
            if len(examples) < 4:
                bad = steps[o - 1]
                examples.append((iid, bad[:95]))
        elif n is not None:
            only_new += 1
print(f"  chains flagged by the OLD checker: {old_flagged}")
print(f"  chains flagged by the FIXED checker: {new_flagged}")
print(f"    both agree            : {both}")
print(f"    OLD only (FALSE ALARM): {only_old}  "
      f"= {100*only_old/max(old_flagged,1):.1f}% of old flags")
print(f"    NEW only (old missed) : {only_new}")
print("  examples the old checker wrongly accused:")
for iid, text in examples:
    print(f"    {iid}: {text}")

print()
print("=" * 68)
print("FIX 2 — diverse-8 share distortion")
print("=" * 68)
import copy
old_pools = s17.apply_diverse_views(copy.deepcopy(raw))
new_pools = diverse8_fixed.apply_diverse_views(copy.deepcopy(raw))

flips = plur_changes = 0
old_ls = new_ls = 0
for iid in old_pools:
    o, n = old_pools[iid], new_pools[iid]
    o_land, n_land = o["share"] >= 0.70, n["share"] >= 0.70
    old_ls += o_land
    new_ls += n_land
    if o_land != n_land:
        flips += 1
    if not match(o["plurality"], n["plurality"]):
        plur_changes += 1
print(f"  pools: {len(old_pools)}")
print(f"  landslide pools  OLD {old_ls}  ->  FIXED {new_ls}  "
      f"(status flips: {flips})")
print(f"  plurality changes: {plur_changes}")

for label, pools in (("OLD (distorted share)", old_pools),
                     ("FIXED (true share)", new_pools)):
    world = World(pools)
    state, contested, osc = converge(pools, world)
    m = metrics(pools, state, contested)
    print(f"\n  {label}")
    print(f"    contested {len(contested)}   oscillators {len(osc)}")
    print(f"    coverage {m['coverage']:.4f}  precision {m['precision']:.4f}"
          f"  contested_acc {m['contested_accuracy']:.4f}")
print("\nDONE")
