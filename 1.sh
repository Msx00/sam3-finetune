#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/afs/zhemin/zjx/Project/mysam/MedSAM3-main"
CONDA_SH="/mnt/afs/zhemin/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="sam3"
CONFIG="${PROJECT_ROOT}/configs/moe_sam3.yaml"

# Manifest containing the complete training-patient selection.
SOURCE_MANIFEST="${PROJECT_ROOT}/outputs/hierarchical_moe_sam3/selected_patients.json"

# Reduced manifest used only to estimate Boundary thresholds.
SUBSET_MANIFEST="${PROJECT_ROOT}/outputs/hierarchical_moe_sam3/selected_patients_boundary_subset.json"

# Boundary threshold file consumed by training.
OUTPUT="${PROJECT_ROOT}/outputs/hierarchical_moe_sam3/boundary_thresholds.json"

# Number of patients sampled for Boundary threshold estimation.
NUM_MR_PATIENTS=50
NUM_US_PATIENTS=50
SAMPLING_SEED=42

if [[ ! -d "${PROJECT_ROOT}" ]]; then
    echo "Error: project directory not found: ${PROJECT_ROOT}"
    exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Error: conda initialization script not found: ${CONDA_SH}"
    exit 1
fi

cd "${PROJECT_ROOT}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0
export CC="/mnt/afs/zhemin/miniconda3/envs/sam3/bin/x86_64-conda-linux-gnu-gcc"
export CXX="/mnt/afs/zhemin/miniconda3/envs/sam3/bin/x86_64-conda-linux-gnu-g++"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${PROJECT_ROOT}/outputs/hierarchical_moe_sam3/logs"
LOG_FILE="${PROJECT_ROOT}/outputs/hierarchical_moe_sam3/logs/compute_boundary_thresholds_subset_$(date +%Y%m%d_%H%M%S).log"

# Display all output in the terminal and save it to a timestamped log file.
exec > >(tee "${LOG_FILE}") 2>&1

echo "Project root: ${PROJECT_ROOT}"
echo "Conda environment: ${CONDA_ENV}"
echo "Python: $(command -v python)"
echo "Log file: ${LOG_FILE}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "Error: config not found: ${CONFIG}"
    exit 1
fi

if [[ ! -f "${SOURCE_MANIFEST}" ]]; then
    echo "Error: selected patient manifest not found:"
    echo "${SOURCE_MANIFEST}"
    exit 1
fi

python - \
    "${SOURCE_MANIFEST}" \
    "${SUBSET_MANIFEST}" \
    "${NUM_MR_PATIENTS}" \
    "${NUM_US_PATIENTS}" \
    "${SAMPLING_SEED}" <<'PY'
import json
import random
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
num_mr = int(sys.argv[3])
num_us = int(sys.argv[4])
seed = int(sys.argv[5])

with source_path.open("r", encoding="utf-8") as handle:
    source = json.load(handle)

# Support both the current manifest fields and the legacy nested fields.
mr_ids = source.get(
    "mr_patient_ids",
    source.get("patients", {}).get("MR", []),
)
us_ids = source.get(
    "us_patient_ids",
    source.get("patients", {}).get("US", []),
)

mr_ids = sorted({int(patient_id) for patient_id in mr_ids})
us_ids = sorted({int(patient_id) for patient_id in us_ids})

if num_mr > len(mr_ids):
    raise ValueError(
        f"Requested {num_mr} MR patients, but only {len(mr_ids)} are available"
    )

if num_us > len(us_ids):
    raise ValueError(
        f"Requested {num_us} US patients, but only {len(us_ids)} are available"
    )

selected_mr = sorted(random.Random(seed).sample(mr_ids, num_mr))
selected_us = sorted(random.Random(seed).sample(us_ids, num_us))

subset = {
    "sampling_mode": "random_boundary_threshold_subset",
    "seed": seed,
    "source_manifest": str(source_path.resolve()),
    "mr_patient_ids": selected_mr,
    "us_patient_ids": selected_us,
    "num_mr_patients": len(selected_mr),
    "num_us_patients": len(selected_us),
    "total_patients": len(selected_mr) + len(selected_us),
}

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as handle:
    json.dump(subset, handle, ensure_ascii=False, indent=2)

print(f"Saved subset manifest: {output_path.resolve()}")
print(f"MR patients ({len(selected_mr)}): {selected_mr}")
print(f"US patients ({len(selected_us)}): {selected_us}")
print(f"Total patients: {len(selected_mr) + len(selected_us)}")
PY

echo
echo "Starting Boundary threshold calculation..."
echo "Config: ${CONFIG}"
echo "Selected patients: ${SUBSET_MANIFEST}"
echo "Output: ${OUTPUT}"
echo

python -u tools/compute_boundary_thresholds.py \
    --config "${CONFIG}" \
    --selected-patients "${SUBSET_MANIFEST}" \
    --output "${OUTPUT}"

echo
echo "Boundary threshold calculation completed."
echo "Result: ${OUTPUT}"

python -m json.tool "${OUTPUT}"
