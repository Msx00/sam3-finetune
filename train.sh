#!/usr/bin/env bash
set -euo pipefail

# Resolve code relative to this script so the repository can be cloned to any
# server directory. Dataset/checkpoint paths remain defined by the YAML config.
#
# Default run: start from checkpoint/sam3.pt and train every module group
# jointly from scratch. No stage1-4 checkpoint is loaded and no curriculum is
# applied. STAGE is intentionally left empty so the stage used by the code is
# taken from the YAML (training.stage); set STAGE=<n> only if you deliberately
# want a different stage.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}/MedSAM3-main}"
CONDA_SH="${CONDA_SH:-/mnt/afs/zhemin/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sam3}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/moe_sam3_train_from_scratch.yaml}"
GPU_IDS="${GPU_IDS:-0}"
RESUME="${RESUME:-}"
STAGE="${STAGE:-}"

[[ -d "${PROJECT_ROOT}" ]] || { echo "Project not found: ${PROJECT_ROOT}"; exit 1; }
[[ -f "${CONDA_SH}" ]] || { echo "Conda init not found: ${CONDA_SH}"; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}"; exit 1; }
if [[ -n "${RESUME}" && ! -f "${RESUME}" ]]; then
  echo "Resume checkpoint not found: ${RESUME}"
  exit 1
fi

cd "${PROJECT_ROOT}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
DEFAULT_CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
DEFAULT_CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
[[ ! -x "${DEFAULT_CC}" ]] || export CC="${CC:-${DEFAULT_CC}}"
[[ ! -x "${DEFAULT_CXX}" ]] || export CXX="${CXX:-${DEFAULT_CXX}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a DEVICE_ARGS <<< "${GPU_IDS}"
if [[ ${#DEVICE_ARGS[@]} -eq 0 ]]; then
  echo "GPU_IDS must contain at least one CUDA device index"
  exit 1
fi

LOG_DIR="${PROJECT_ROOT}/outputs/train_from_scratch/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

COMMAND=(
  python -u train_moe_sam3.py
  --config "${CONFIG}"
  --device "${DEVICE_ARGS[@]}"
)
if [[ -n "${STAGE}" ]]; then
  COMMAND+=(--stage "${STAGE}")
fi
if [[ -n "${RESUME}" ]]; then
  COMMAND+=(--resume "${RESUME}")
fi

echo "Config: ${CONFIG}"
echo "GPU IDs: ${GPU_IDS}"
echo "Stage: ${STAGE:-<from config>}"
echo "Resume: ${RESUME:-disabled}"
echo "Log: ${LOG_FILE}"
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
