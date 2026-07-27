

cd /mnt/afs/zhemin/zjx/Project/mysam/MedSAM3-main
source /mnt/afs/zhemin/miniconda3/etc/profile.d/conda.sh
conda activate sam3

export HYDRA_FULL_ERROR=1


export CC=/mnt/afs/zhemin/miniconda3/envs/sam3/bin/x86_64-conda-linux-gnu-gcc
export CXX=/mnt/afs/zhemin/miniconda3/envs/sam3/bin/x86_64-conda-linux-gnu-g++

mkdir -p outputs/hierarchical_moe_sam3/logs
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

LOG_FILE="outputs/hierarchical_moe_sam3/logs/train_stage5_$(date +%Y%m%d_%H%M%S).log"

set -o pipefail

python -u train_moe_sam3.py \
  --config configs/moe_sam3.yaml \
  --device 0 1 \
  --stage 4 \
  2>&1 | tee "$LOG_FILE"