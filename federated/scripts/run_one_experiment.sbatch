#!/bin/bash
#SBATCH --job-name=fed_env
#SBATCH --output=logs/fed_env_%A_%a.out
#SBATCH --error=logs/fed_env_%A_%a.err
#SBATCH --time=100:00:00
#SBATCH --partition=gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=1-144%8

# ============================================================
# Slurm array script for federated experiments
# 144 jobs:
# 4 models × 2 datasets × 3 methods × 3 clients × 2 partitions
# ============================================================

set -e

echo "============================================================"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_JOB_ID: ${SLURM_ARRAY_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "HOSTNAME: $(hostname)"
echo "START TIME: $(date)"
echo "============================================================"

# ----------------------------
# 1. Project paths
# ----------------------------
PROJECT_ROOT="/gpfs/work/aac/yuxuanguo23/iot_fed_project/Federated-Edge-AI-for-Real-Time-Environmental-Anomaly-Detection-in-IoT-Networks"
FED_DIR="${PROJECT_ROOT}/Edge-Machine-Learning-Models/federated"
JOB_LIST="${FED_DIR}/scripts/job_list.txt"

cd "${FED_DIR}"

mkdir -p logs
mkdir -p results

# ----------------------------
# 2. Environment setup
# ----------------------------
# module purge
# module load cuda/11.8
# module load cudnn/8.6
# module load python/3.10

source ~/.bashrc
conda activate federated_env

# ----------------------------
# 3. TensorFlow runtime options
# ----------------------------
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=1

# Optional: avoid grabbing all GPU memory
export TF_FORCE_GPU_ALLOW_GROWTH=true

# ----------------------------
# 4. Get command from job list
# ----------------------------
CMD_ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOB_LIST}")

echo "Command args:"
echo "${CMD_ARGS}"

echo "Running:"
echo "python run_experiment.py ${CMD_ARGS}"

python run_experiment.py ${CMD_ARGS}

echo "============================================================"
echo "END TIME: $(date)"
echo "DONE"
echo "============================================================"