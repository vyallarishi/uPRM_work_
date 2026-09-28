# Results

All numbers are AUC (0.5 = chance) unless stated. "Contaminated" chains are
wrong chains that agree with a committed answer, i.e. what would enter the
training set with a wrong label. The files are in `data/uPRM_work_/recorded_results/`, outside git.

## Step scoring on a small balanced set (phaseA, 15 languages × 150 problems)

| file | script | result |
|---|---|---|
| `stage32_results.json` | `step_scoring/toy1_first_broken_step` | clean vs broken **0.835** (700 chains); localisation of the first false equation 17.6% exact vs 21.7% chance |
| (log) | `step_scoring/length_confound_check` | length matched: **0.891**; length alone 0.448, so length works *against* the result |
| `stage33_results.json` | `step_scoring/toy2_joint_scoring` | independent 0.847 vs joint 0.848 (length matched): joint scoring adds nothing; poisoned neighbours 0.724 |
| `stage42_results.json`, `stage42_coverage.json` | `step_scoring/paper_faithful_rerun` | the paper's absorbing-error prompt: 0.816, localisation lift −4.1 pts (worse); newline vs fallback step splitting: identical |
| `stage34_results.json` | `at_scale/harvest_cleaning_small` | genuine vs contaminated 0.742 (0.809 matched); on the residue where the world vote is also fooled: 0.934 (n=287) |

## At scale (37 languages × 1,500 problems, 344,383 chains scored)

| file | script | result |
|---|---|---|
| `stage36_results.json` | `at_scale/decide_with_scores` | contaminated-chain AUC **0.646** (world agreement alone: 0.733). Tie-break inside the loop: fixed 431 pools, broke 3,967 (precision 94.45 → 88.03). uPRM alone: precision 86.89 |
| `stage37R_results.json` | `at_scale/error_types_fixed_arithmetic` | why it collapsed: 0.600 on wrong chains that won a landslide vs 0.848 on those that did not. Error type (arithmetic slip or not) makes no difference: 0.635 vs 0.648 |
| `stage38_results.json` | `at_scale/pool_level_features` | pool-level "is this commit wrong?": best PRM feature 0.734; vote share alone 0.792. Adds nothing beyond voting |
| `stage39R_results.json` | `at_scale/sft_chain_selection_fixed_arithmetic` | choosing which winning chain to train on: ranks clean above lucky-guess chains in 73.0% of 515 pools; globally 0.528 |
| `stage40R_results.json` | `at_scale/length_corrected_selection_fixed_arithmetic` | length correction lifts that to 76.9%; the selector still prefers short chains |
| `stage43_results_full.json` | `step_coherence/step_coherence` | is the error position coordinated across chains of the same problem? Yes, weakly: exact-position agreement 0.393 vs 0.354 null; binary agreement 0.888 vs 0.805 |

`stage37/39/40_results.json` (without R) are the first runs, computed with
the arithmetic checker that ignored operator precedence; the `R` files are
the reruns with the fixed checker (`ICM_work_/icm/experiments/q07_audits/fix_arithmetic_precedence.py`).
The bug inverted one finding (error-type gap +0.067 → −0.014) and shrank the
selector's eligible pools from 2,619 to 515.

## Caveats worth knowing

- Qwen3-14B-Base is a base model; uPRM used an instruction-tuned grader.
- Steps are split on newlines. 17.2% of chains yield too few steps and were
  dropped from every PRM measurement, unevenly by language (Malayalam 34.5%,
  English 6.7%). `q02_does_it_help_at_scale/step_partition_by_language.py`.
- Gold is used only to score, never in prompts or decisions.
