#!/usr/bin/env bash
set -euo pipefail

# Build every artifact train.sh loads (COCO/box JSONs, patient manifests and the
# per-slice area/boundary label cache) once, ahead of training.  Same
# environment conventions as train.sh: paths live in the YAML config, the conda
# environment is only used to run the Python below.
#
# Examples:
#   ./run_preprocess.sh                          # all stages, default config
#   ./run_preprocess.sh verify                   # coverage report only
#   ./run_preprocess.sh labels --workers 64      # refresh just the label cache
#   STAGE=all CONFIG=/path/to/config.yaml ./run_preprocess.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONDA_SH="${CONDA_SH:-/mnt/afs/zhemin/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sam3}"
CONFIG="${CONFIG:-${REPO_ROOT}/MedSAM3-main/configs/moe_sam3_train_from_scratch.yaml}"
WORKERS="${WORKERS:-}"
LOG_DIR="${SCRIPT_DIR}/logs"

[[ -f "${CONDA_SH}" ]] || { echo "Conda init not found: ${CONDA_SH}"; exit 1; }
[[ -f "${CONFIG}" ]] || { echo "Config not found: ${CONFIG}"; exit 1; }

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export PYTHONPATH="${REPO_ROOT}/MedSAM3-main:${PYTHONPATH:-}"

ARGS=(--config "${CONFIG}")
[[ -z "${WORKERS}" ]] || ARGS+=(--workers "${WORKERS}")
if [[ $# -eq 0 ]]; then
  set -- all
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/preprocess_$(date +%Y%m%d_%H%M%S).log"

echo "Config: ${CONFIG}"
echo "Stage : $*"
echo "Log   : ${LOG_FILE}"
python -u "${SCRIPT_DIR}/preprocess.py" "${ARGS[@]}" "$@" 2>&1 | tee "${LOG_FILE}"
