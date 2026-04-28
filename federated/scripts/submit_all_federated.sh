#!/bin/bash

set -e

PROJECT_ROOT="/gpfs/work/aac/yuxuanguo23/iot_fed_project/Federated-Edge-AI-for-Real-Time-Environmental-Anomaly-Detection-in-IoT-Networks"
FED_DIR="${PROJECT_ROOT}/Edge-Machine-Learning-Models/federated"
SCRIPT_DIR="${FED_DIR}/scripts"
RESULT_DIR="${FED_DIR}/results"
JOB_LIST="${SCRIPT_DIR}/job_list.txt"

mkdir -p "${FED_DIR}/logs"
mkdir -p "${RESULT_DIR}"

cd "${FED_DIR}"

echo "Generating job list..."

python "${SCRIPT_DIR}/make_job_list.py" \
  --output "${JOB_LIST}" \
  --project_root "${PROJECT_ROOT}" \
  --result_dir "${RESULT_DIR}" \
  --rounds 100 \
  --local_epochs 1 \
  --seed 2024 \
  --alpha 0.5 \
  --patience 10

N_JOBS=$(wc -l < "${JOB_LIST}")

echo "Total jobs: ${N_JOBS}"

# Update array range dynamically if needed
# Current sbatch file has #SBATCH --array=1-144%8
# If N_JOBS is not 144, submit with command-line override:
sbatch --array=1-${N_JOBS}%8 "${SCRIPT_DIR}/run_one_experiment.sbatch"

echo "Submitted ${N_JOBS} jobs."