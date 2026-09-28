#!/usr/bin/env bash
#SBATCH --account=rrg-zhijing
#SBATCH --job-name=stage33-uprm-toy2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=124G
#SBATCH --time=5:00:00
#SBATCH --output=/home/rvyalla/multilingual_icm/data/uPRM_work_/logs/%x-%j.out
set -Eeuo pipefail
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
python -m uprm.experiments.q01_does_step_scoring_work.toy2_joint_scoring --self-test
python -m uprm.experiments.q01_does_step_scoring_work.toy2_joint_scoring --dry-run
python -m uprm.experiments.q01_does_step_scoring_work.toy2_joint_scoring
