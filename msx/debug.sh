#!/usr/bin/env bash

cd /mnt/afs/zhemin/zjx/Project/mysam/MedSAM3-main

source /mnt/afs/zhemin/miniconda3/etc/profile.d/conda.sh
conda activate sam3

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export SVANET_DEBUG_MODE=train_bn_eval
export SVANET_DEBUG_MAX_BATCHES=0

USE_WANDB="${USE_WANDB:-1}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-HMOE-SAM3}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-debug-stage4}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TAGS="${WANDB_TAGS:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-./wandb}"
WANDB_LOG_INTERVAL="${WANDB_LOG_INTERVAL:-10}"
WANDB_WATCH_MODEL="${WANDB_WATCH_MODEL:-0}"
WANDB_RESUME="${WANDB_RESUME:-allow}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

WANDB_ARGS=()
if [[ "$USE_WANDB" == "1" ]]; then
  WANDB_ARGS+=(
    --use-wandb
    --wandb-project "$WANDB_PROJECT"
    --wandb-name "$WANDB_NAME"
    --wandb-mode "$WANDB_MODE"
    --wandb-dir "$WANDB_DIR"
    --wandb-log-interval "$WANDB_LOG_INTERVAL"
    --wandb-resume "$WANDB_RESUME"
  )
  [[ -n "$WANDB_ENTITY" ]] && WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  [[ -n "$WANDB_GROUP" ]] && WANDB_ARGS+=(--wandb-group "$WANDB_GROUP")
  [[ -n "$WANDB_RUN_ID" ]] && WANDB_ARGS+=(--wandb-run-id "$WANDB_RUN_ID")
  [[ "$WANDB_WATCH_MODEL" == "1" ]] && WANDB_ARGS+=(--wandb-watch-model)
  if [[ -n "$WANDB_TAGS" ]]; then
    IFS=',' read -r -a WANDB_TAG_ARRAY <<< "$WANDB_TAGS"
    WANDB_ARGS+=(--wandb-tags "${WANDB_TAG_ARRAY[@]}")
  fi
fi

python -u train_moe_sam3.py \
  --config configs/moe_sam3_stage4_debug.yaml \
  --device 0 \
  --stage 4 \
  "${WANDB_ARGS[@]}" \
  2>&1 | tee outputs/debug_stage4_3epoch.log
