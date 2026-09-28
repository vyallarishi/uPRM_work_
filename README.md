# uPRM: step-level scoring of reasoning chains

Can the model find the **first wrong step** in one of its own solutions?

The ICM pipeline ([ICM_work_](../ICM_work_)) labels whole answers by letting
37 languages vote. It never judges a chain's *reasoning*. uPRM (arXiv
2605.10158) offers a way to: after each step the model says `+` or `-`, and
from those marker probabilities we get, per chain, the probability that no
step is wrong (`p_clean`) and the most likely position of the first error.

**What we found**

- On a balanced set of clean vs broken chains it works: AUC 0.835, 0.891
  after length matching. The first formulation in the project to clear the
  0.70 bar.
- It cannot say *where* the error is: localisation is at chance (17.6% vs
  21.7%), and the paper's own prompt makes it worse.
- The paper's joint scoring (many chains in one context) adds nothing, and
  poisoning the neighbours' verdicts drags it down (herding).
- **At scale it collapses to 0.646.** The hard cases are wrong chains that
  won a landslide vote: on those it scores 0.600, against 0.848 on other
  wrong chains. Using it inside the pipeline broke 3,967 pools to fix 431.
- Its per-step `+/-` vectors are what the probe work
  ([probe_work_](../probe_work_)) builds on.

## Start here

1. [uprm/scoring.py](uprm/scoring.py): the scorer in four functions.
2. [recorded_results/RESULTS.md](recorded_results/RESULTS.md): every number, with the file that produced it.
3. [uprm/experiments/](uprm/experiments/): the experiments, three folders, each with a README.

## What is where

| folder | what it is |
|---|---|
| [uprm/scoring.py](uprm/scoring.py) | the step scorer (split steps, build the grading transcript, Eq. 6) |
| [uprm/experiments/](uprm/experiments/) | every experiment, grouped: `q01_does_step_scoring_work/`, `q02_does_it_help_at_scale/`, `q03_is_the_error_position_coordinated/` |
| [recorded_results/RESULTS.md](recorded_results/RESULTS.md) | what every result file shows; the files and the CPU-run logs are in `data/uPRM_work_/recorded_results/`, outside git |
| [tests/](tests/) | proves `uprm/scoring.py` equals the experiment code |
| [tools/](tools/) | checks that the experiment files are unchanged from the originals |
| `data/uPRM_work_/runs/` | caches and re-run outputs (not in git, 165 MB) |

## Running things

This repository imports the ICM one. Keep the two checkouts side by side
(`uprm/__init__.py` finds `../ICM_work_` by itself) or `pip install -e ../ICM_work_`.

```bash
python -m unittest discover -s tests        # scoring.py == experiment code (1 s)
python -m uprm.experiments.q01_does_step_scoring_work.toy1_first_broken_step --self-test
```

Every `.sbatch` job runs from `data/uPRM_work_/runs/` with both repositories on `PYTHONPATH`.
Scoring needs the GPU and the generated pools on `/scratch`; the CPU analyses
(`q02_does_it_help_at_scale/decide_with_scores.py` and the ones after it) need only the caches
in `data/uPRM_work_/runs/`.

## Where this came from

The experiment files are copies of `~/phase0/stageNN_*.py`, renamed and
grouped; the code is unchanged apart from import lines and output paths
(`tools/check_provenance.py` verifies this). `uprm/provenance.json` maps every
file to its original name.
