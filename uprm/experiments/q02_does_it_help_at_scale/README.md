# at_scale: does step scoring help the pipeline?

Data: `qwen3-14b-full37-1500` (37 languages × 1,500 problems). Every chain was
scored once (`score_all_chains.py`, 4 GPU shards, `data/uPRM_work_/runs/stage35_scores_*of4.jsonl`);
everything after that is CPU work on the cache.

| file | was | what it does | result |
|---|---|---|---|
| `harvest_cleaning_small.py` | stage34_uprm_harvest | on phaseA: can `p_clean` separate genuine from contaminated training chains? | 0.742 overall; 0.934 where the world vote is also fooled (n=287) |
| `score_all_chains.py` | stage35_score_all | **the scoring pass**: 344,383 chains, decides nothing | the cache |
| `decide_with_scores.py` | stage36_decide | replay every use of the score against the shipped pipeline: A clean the training set, B break ties in the loop, C uPRM alone | A 0.646; B fixed 431 / broke 3,967; C precision 86.9% |
| `error_types.py`, `error_types_fixed_arithmetic.py` | stage37, stage37R | why 0.891 became 0.646: error type, or harder negatives? | harder negatives: landslide winners 0.600 vs others 0.848 |
| `pool_level_features.py` | stage38_pool_level | flag a *pool* as untrustworthy from the scores of its winning chains | best 0.734; vote share alone 0.792 |
| `sft_chain_selection.py`, `sft_chain_selection_fixed_arithmetic.py` | stage39, stage39R | which winning chain to train on: does it rank clean above lucky-guess chains? | 73.0% of 515 pools |
| `length_corrected_selection.py`, `length_corrected_selection_fixed_arithmetic.py` | stage40, stage40R | the selector prefers short chains; four length corrections | best 76.9% |
| `step_partition_by_language.py` | step_partition_check | do newlines split steps in every language? | no: 36% of Kazakh chains are one line vs 7.6% of English |
| `fix_impact.py` | fix_impact | what the two audit fixes (arithmetic precedence, pool view) change on real data | see `recorded_results/logs/fix_impact.log` |
| `audit_arithmetic_false_positives.py` | audit_arith | the old checker's false alarms | 66.7% of its flags |
| `audit_precedence_bug_impact.py` | audit_impact | the bug's effect on stage 37/39/40 | error-type gap inverts; eligible pools 2,619 → 515 |
| `audit_arm_b.py`, `audit_arm_b_counts.py` | audit_armb, audit_armb2 | the 431 / 3,967 counts, recomputed | confirmed |

The `_fixed_arithmetic` files are reruns after the operator-precedence bug
in the arithmetic checker was found (the checker is
`ICM_work_/icm/experiments/q07_audits/fix_arithmetic_precedence.py`). Both
versions are kept so every recorded number stays reproducible.
