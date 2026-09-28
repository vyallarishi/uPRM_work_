# uprm/experiments

Three folders, in the order the work happened.

| # | folder | question | answer |
|---|---|---|---|
| 1 | [q01_does_step_scoring_work/](q01_does_step_scoring_work/) | Does step-level scoring separate clean from broken chains, and can it find the broken step? | Separates: 0.835 / 0.891. Cannot localise. Joint scoring adds nothing. |
| 2 | [q02_does_it_help_at_scale/](q02_does_it_help_at_scale/) | Does it help the pipeline: cleaning training data, breaking ties, choosing chains? | No. 0.646 on the chains that matter; breaks more pools than it fixes. |
| 3 | [q03_is_the_error_position_coordinated/](q03_is_the_error_position_coordinated/) | Is the error position coordinated across chains, even if it is not correct? | Weakly yes. |

## Conventions

- File names say what the experiment tested; `uprm/provenance.json` maps
  each one to its original `~/phase0/stageNN_*.py` name.
- A `.sbatch` next to a `.py` is the cluster job that ran it (GPU). The
  `q02_does_it_help_at_scale/` analyses after scoring have no job: they ran on CPU and their
  printed output is in `recorded_results/logs/`.
- Jobs `cd` into `data/uPRM_work_/runs/` and put this repo and `ICM_work_` on `PYTHONPATH`.
- Files import the ICM experiment code as `icm.experiments…` (the pools,
  the voting loop, the arithmetic checker). Nothing is duplicated.
- The code is unchanged from the originals apart from import lines and
  output paths (`tools/check_provenance.py`).
