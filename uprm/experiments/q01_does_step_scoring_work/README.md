# step_scoring: does the model know which step is wrong?

Data: `qwen3-14b-phaseA` (15 languages × 150 problems), 700 chains balanced
between clean and broken.

| file | was | what it does | result |
|---|---|---|---|
| `toy1_first_broken_step.py` | stage32_uprm_toy1 | **The scorer.** Per step, read P(`+`) vs P(`-`); `p_clean` = P(no wrong step). A: clean vs broken AUC. B: does the most likely error position hit the first false `<<a op b=c>>`? | A 0.835. B 17.6% vs 21.7% chance |
| `length_confound_check.py` | uprm_confound | is 0.835 a length artifact? (longer chains sum more log-probs) | no: length matched 0.891, length alone 0.448 |
| `toy2_joint_scoring.py` | stage33_uprm_toy2 | the paper's trick: score 8 chains in one context. Arms: independent, joint, shuffled (other problems' chains), poisoned (neighbours marked `-`) | joint = independent (0.848 vs 0.847); poisoned 0.724 |
| `paper_faithful_rerun.py` | stage42_uprm_faithful | two deviations from the paper fixed separately: the "absorbing error" prompt clause, and step splitting for chains without newlines | prompt clause makes localisation worse (−4.1 pts); splitting changes nothing |
| `per_step_label_vectors.py` | stage44_perstep_score | rescore every chain keeping one `+`/`-` per step (a vector, not a single position). Output `data/uPRM_work_/runs/stage44_perstep_allplus_0of4.jsonl` is the input of the probe repository's data layer | 49 MB of labels |
| `audit_eq6_indexing.py`, `audit_toy1_numbers.py`, `audit_toy2_subsets.py`, `audit_toy2_arms.py` | audit_eq6, audit_toy, audit_toy2, audit_toy2b | the review: Eq. 6 indexing correct; the all-plus prefix shortcut is exact; toy-2 arms were compared on different chain subsets | |

Jobs: `toy1_first_broken_step.sbatch`, `toy2_joint_scoring.sbatch`,
`paper_faithful_rerun.sbatch`, `per_step_label_vectors.sbatch` (GPU).
