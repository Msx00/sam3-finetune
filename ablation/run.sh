#!/usr/bin/env bash
set -euo pipefail

# Resolve code relative to this script. The ablation runner still inherits all
# dataset/checkpoint paths from the canonical Stage-5 configuration.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-${WORKSPACE_ROOT}/MedSAM3-main}"
ABLATION_ROOT="${ABLATION_ROOT:-${SCRIPT_DIR}}"
CONDA_SH="${CONDA_SH:-/mnt/afs/zhemin/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-sam3}"

STUDY="${STUDY:-${ABLATION_ROOT}/study.yaml}"
SUITE="${SUITE:-primary}"
DEVICE_SETS="${DEVICE_SETS:-0,1}"
SEEDS="${SEEDS:-}"
INCLUDE="${INCLUDE:-}"
MAX_PARALLEL="${MAX_PARALLEL:-}"
MASTER_PORT="${MASTER_PORT:-29600}"
DRY_RUN="${DRY_RUN:-0}"
RERUN_COMPLETED="${RERUN_COMPLETED:-0}"
FAIL_FAST="${FAIL_FAST:-1}"

[[ -d "${PROJECT_ROOT}" ]] || { echo "Project not found: ${PROJECT_ROOT}"; exit 1; }
[[ -d "${ABLATION_ROOT}" ]] || { echo "Ablation directory not found: ${ABLATION_ROOT}"; exit 1; }
[[ -f "${ABLATION_ROOT}/run_ablation.py" ]] || { echo "Runner not found: ${ABLATION_ROOT}/run_ablation.py"; exit 1; }
[[ -f "${STUDY}" ]] || { echo "Study not found: ${STUDY}"; exit 1; }
[[ -f "${CONDA_SH}" ]] || { echo "Conda init not found: ${CONDA_SH}"; exit 1; }

cd "${WORKSPACE_ROOT}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
DEFAULT_CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
DEFAULT_CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
[[ ! -x "${DEFAULT_CC}" ]] || export CC="${CC:-${DEFAULT_CC}}"
[[ ! -x "${DEFAULT_CXX}" ]] || export CXX="${CXX:-${DEFAULT_CXX}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

read -r -a DEVICE_ARGS <<< "${DEVICE_SETS}"
[[ ${#DEVICE_ARGS[@]} -gt 0 ]] || { echo "DEVICE_SETS must not be empty"; exit 1; }

RUN_ARGS=(
  python -u "${ABLATION_ROOT}/run_ablation.py" run
  --study "${STUDY}"
  --suite "${SUITE}"
  --device-sets "${DEVICE_ARGS[@]}"
  --master-port "${MASTER_PORT}"
  --logs-dir "${ABLATION_ROOT}/logs"
  --state-dir "${ABLATION_ROOT}/state"
)
VALIDATE_ARGS=(
  python -u "${ABLATION_ROOT}/run_ablation.py" validate
  --study "${STUDY}"
  --suite "${SUITE}"
)

if [[ -n "${SEEDS}" ]]; then
  read -r -a SEED_ARGS <<< "${SEEDS}"
  RUN_ARGS+=(--seeds "${SEED_ARGS[@]}")
  VALIDATE_ARGS+=(--seeds "${SEED_ARGS[@]}")
fi
if [[ -n "${INCLUDE}" ]]; then
  RUN_ARGS+=(--include "${INCLUDE}")
  VALIDATE_ARGS+=(--include "${INCLUDE}")
fi
if [[ -n "${MAX_PARALLEL}" ]]; then
  RUN_ARGS+=(--max-parallel "${MAX_PARALLEL}")
fi
[[ "${DRY_RUN}" == "1" ]] && RUN_ARGS+=(--dry-run)
[[ "${RERUN_COMPLETED}" == "1" ]] && RUN_ARGS+=(--rerun-completed)
[[ "${FAIL_FAST}" == "1" ]] && RUN_ARGS+=(--fail-fast)

echo "Study: ${STUDY}"
echo "Suite: ${SUITE}"
echo "Device sets: ${DEVICE_SETS}"
echo "Seeds: ${SEEDS:-study defaults}"
echo "Dry run: ${DRY_RUN}"

"${VALIDATE_ARGS[@]}"
"${RUN_ARGS[@]}"
