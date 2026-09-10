#!/usr/bin/env bash
set -euo pipefail

# Server paths are intentionally absolute: this entry is executed on the
# training server, not from the local checkout used to edit the project.
PROJECT_ROOT="/mnt/afs/zhemin/zjx/Project/mysam/MedSAM3-main"
CONDA_SH="/mnt/afs/zhemin/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sam3"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/moe_sam3_stage5_direct.yaml}"
GPU_IDS="${GPU_IDS:-0 1}"
RESUME="${RESUME:-}"

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
export CC="/mnt/afs/zhemin/miniconda3/envs/sam3/bin/x86_64-conda-linux-gnu-gcc"
export CXX="/mnt/afs/zhemin/miniconda3/envs/sam3/bin/x86_64-conda-linux-gnu-g++"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a DEVICE_ARGS <<< "${GPU_IDS}"
if [[ ${#DEVICE_ARGS[@]} -eq 0 ]]; then
  echo "GPU_IDS must contain at least one CUDA device index"
  exit 1
fi

LOG_DIR="${PROJECT_ROOT}/outputs/stage5_direct_noise/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_v2_$(date +%Y%m%d_%H%M%S).log"

COMMAND=(
  python -u train_moe_sam3.py
  --config "${CONFIG}"
  --device "${DEVICE_ARGS[@]}"
  --stage 5
)
if [[ -n "${RESUME}" ]]; then
  COMMAND+=(--resume "${RESUME}")
fi

echo "Config: ${CONFIG}"
echo "GPU IDs: ${GPU_IDS}"
echo "Resume: ${RESUME:-disabled}"
echo "Log: ${LOG_FILE}"
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
