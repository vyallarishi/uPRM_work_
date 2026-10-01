#!/usr/bin/env bash
# What the two audit fixes change on the real 37x1500 data (CPU, no model):
#   1. the precedence-correct arithmetic checker vs the old one, over every chain
#   2. the fixed pool view vs the old one: landslide/contested counts and the
#      converged coverage/precision -- the number the scale run needs before
#      it adopts the fix.
# Also exercises the uPRM -> ICM cross-repo imports on real data.
# CPU only -> def-zhijing (rrg-zhijing rejects CPU jobs).
#SBATCH --account=def-zhijing
#SBATCH --job-name=uprm-fix-impact
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/uPRM_work_/logs/%x-%j.out
set -Eeuo pipefail
module reset
module load python/3.11
PROJECT_STORAGE="${PROJECT:-${HOME}}"
source "${PROJECT_STORAGE}/venvs/multilingual-icm/bin/activate"
cd /home/rvyalla/multilingual_icm/data/uPRM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/uPRM_work_:/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"

python -u -m uprm.experiments.q02_does_it_help_at_scale.fix_impact | tee fix_impact.rerun.log
echo "== recorded run (2026-08-29) for comparison:"
cat /home/rvyalla/multilingual_icm/data/uPRM_work_/recorded_results/logs/fix_impact.log
