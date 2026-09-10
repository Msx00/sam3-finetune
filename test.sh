#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}/MedSAM3-main}"
CONDA_SH="${CONDA_SH:-/mnt/afs/zhemin/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sam3}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/test.yaml}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/outputs/stage5_direct_noise/stage5_joint_best.pt}"
DEVICE="${DEVICE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-test_outputs/stage5_direct_noise/prompt_eval_mixed_noise}"

[[ -d "${PROJECT_ROOT}" ]] || { echo "Project not found: ${PROJECT_ROOT}"; exit 1; }
[[ -f "${CONDA_SH}" ]] || { echo "Conda init not found: ${CONDA_SH}"; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}"; exit 1; }
[[ -f "${CHECKPOINT}" ]] || { echo "Checkpoint not found: ${CHECKPOINT}"; exit 1; }

cd "${PROJECT_ROOT}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export CUDA_VISIBLE_DEVICES="${DEVICE}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"
python -u test_moe_sam3_prompts.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --device 0 \
  --stage 5 \
  --data-mode mixed \
  --box-noise-std 0.1 \
  --box-noise-max 10 \
  --box-noise-seed 42 \
  --prompt-modes none text coarse_box box text_box \
  --mr-max-patients 70 \
  --us-max-patients 70 \
  --batch-size 1 \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/evaluation.log"
