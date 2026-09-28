#!/usr/bin/env bash
# Per-step +/- label vectors. Submit ONCE PER SHARD for the allplus arm:
#   for i in 0 1 2 3; do sbatch stage44_perstep.sbatch allplus $i 4; done
# and once for the subsample arm:
#   sbatch stage44_perstep.sbatch selfcons 0 1
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=stage44-perstep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=124G
#SBATCH --time=8:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/uPRM_work_/logs/%x-%j.out
set -Eeuo pipefail
ARM="${1:-allplus}"
SHARD="${2:-0}"
OF="${3:-1}"
module reset
module load python/3.11
module load cuda
PROJECT_STORAGE="${PROJECT:-${HOME}}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source "${PROJECT_STORAGE}/venvs/sm-icm/bin/activate"
cd /home/rvyalla/multilingual_icm/data/uPRM_work_/runs
export PYTHONPATH="/home/rvyalla/multilingual_icm/uPRM_work_:/home/rvyalla/multilingual_icm/ICM_work_${PYTHONPATH:+:$PYTHONPATH}"
python -m uprm.experiments.q01_does_step_scoring_work.per_step_label_vectors --self-test
python -m uprm.experiments.q01_does_step_scoring_work.per_step_label_vectors --arm "$ARM" --shard "$SHARD" --of "$OF" --dry-run
python -m uprm.experiments.q01_does_step_scoring_work.per_step_label_vectors --arm "$ARM" --shard "$SHARD" --of "$OF"
