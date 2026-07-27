#!/usr/bin/env bash
set -Eeuo pipefail

# One-stage end-to-end PEFT for MR/TRUS MoE-SAM3.
# Examples:
#   bash run_end_to_end.sh
#   DEVICES="0 1" CONFIG=configs/moe_sam3.yaml bash run_end_to_end.sh
#   RESUME=outputs/.../stage5_joint_last.pt bash run_end_to_end.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}/MedSAM3-main}"
CONFIG="${CONFIG:-configs/moe_sam3.yaml}"
DEVICES="${DEVICES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="${LOG_DIR:-outputs/hierarchical_moe_sam3_end2end/logs}"
RESUME="${RESUME:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-}"
TRAIN_CASES="${TRAIN_CASES:-30}"
VAL_CASES="${VAL_CASES:-10}"

if (( TRAIN_CASES < 2 || VAL_CASES < 2 )); then
  echo "TRAIN_CASES and VAL_CASES must both be at least 2." >&2
  exit 1
fi

TRAIN_MR_CASES="${TRAIN_MR_CASES:-$((TRAIN_CASES / 2))}"
TRAIN_US_CASES="${TRAIN_US_CASES:-$((TRAIN_CASES - TRAIN_MR_CASES))}"
VAL_MR_CASES="${VAL_MR_CASES:-$((VAL_CASES / 2))}"
VAL_US_CASES="${VAL_US_CASES:-$((VAL_CASES - VAL_MR_CASES))}"

if [[ -n "${CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "CONDA_ENV is set, but conda is not available on PATH." >&2
    exit 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

cd "${PROJECT_DIR}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${PROJECT_DIR}/${CONFIG}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED=1

read -r -a DEVICE_ARGS <<< "${DEVICES}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/train_end_to_end_${TIMESTAMP}.log"

COMMAND=(
  "${PYTHON_BIN}" -u train_moe_sam3.py
  --config "${CONFIG}"
  --device "${DEVICE_ARGS[@]}"
  --end-to-end
  --batch-size "${BATCH_SIZE}"
  --num-mr-train-cases "${TRAIN_MR_CASES}"
  --num-us-train-cases "${TRAIN_US_CASES}"
  --num-mr-val-cases "${VAL_MR_CASES}"
  --num-us-val-cases "${VAL_US_CASES}"
)

if [[ -n "${EPOCHS}" ]]; then
  COMMAND+=(--epochs "${EPOCHS}")
fi

if [[ -n "${RESUME}" ]]; then
  COMMAND+=(--resume "${RESUME}")
fi

# Additional train_moe_sam3.py arguments can be appended to this script.
COMMAND+=("$@")

echo "Project : ${PROJECT_DIR}"
echo "Config  : ${CONFIG}"
echo "Devices : ${DEVICES}"
echo "Batch   : ${BATCH_SIZE} per process"
echo "Train   : ${TRAIN_MR_CASES} MR + ${TRAIN_US_CASES} TRUS cases"
echo "Val     : ${VAL_MR_CASES} MR + ${VAL_US_CASES} TRUS cases"
echo "Log     : ${LOG_FILE}"
printf 'Command :'
printf ' %q' "${COMMAND[@]}"
printf '\n'

set -o pipefail
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
