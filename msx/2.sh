
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/mnt/afs/zhemin/zjx/Project/mysam/MedSAM3-main"
CONFIG="${PROJECT_ROOT}/configs/moe_sam3.yaml"

# 第一阶段生成的420名训练患者清单
SOURCE_MANIFEST="${PROJECT_ROOT}/outputs/hierarchical_moe_sam3/selected_patients.json"

# Boundary阈值计算专用的精简患者清单
SUBSET_MANIFEST="${PROJECT_ROOT}/outputs/hierarchical_moe_sam3/selected_patients_boundary_subset.json"

# 最终Boundary阈值文件，训练时会读取该文件
OUTPUT="${PROJECT_ROOT}/outputs/hierarchical_moe_sam3/boundary_thresholds.json"

# 可根据需要调整
NUM_MR_PATIENTS=50
NUM_US_PATIENTS=50
SAMPLING_SEED=42

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

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

# 兼容当前项目以及旧版manifest字段
mr_ids = source.get(
    "mr_patient_ids",
    source.get("patients", {}).get("MR", [])
)
us_ids = source.get(
    "us_patient_ids",
    source.get("patients", {}).get("US", [])
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

# 分别初始化随机数生成器，确保两个模态抽样可复现
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

python tools/compute_boundary_thresholds.py \
    --config "${CONFIG}" \
    --selected-patients "${SUBSET_MANIFEST}" \
    --output "${OUTPUT}"

echo
echo "Boundary threshold calculation completed."
echo "Result: ${OUTPUT}"

python -m json.tool "${OUTPUT}"
