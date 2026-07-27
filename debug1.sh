cd /mnt/afs/zhemin/zjx/Project/mysam/MedSAM3-main

source /mnt/afs/zhemin/miniconda3/etc/profile.d/conda.sh
conda activate sam3

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SVANET_DEBUG_MODE=forward_bn_eval
export SVANET_DEBUG_MAX_BATCHES=0

python -u train_moe_sam3.py \
  --config configs/moe_sam3_debug.yaml \
  --device 0 \
  --stage 4 \
  2>&1 | tee outputs/debug_stage4_a_full.log