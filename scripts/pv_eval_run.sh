#!/usr/bin/env bash
# Runs eval/eval.py with sensible defaults. Override MODEL/DATASET/MODE/VARIANT
# on the command line, e.g.:
#   MODEL=models/touch_full.ckpt DATASET=labeled-s MODE=zero-shot bash scripts/pv_eval_run.sh
set -euo pipefail

MODEL=${MODEL:-models/vit_speech_only_2.ckpt}
DATASET=${DATASET:-pv}
MODE=${MODE:-linear_probe}
VARIANT=${VARIANT:-Y}
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-4}
VOCAB=${VOCAB:-multimodal/vocab.json}
# Set WANDB=1 to log this run to wandb (otherwise stays local).
WANDB_FLAG=""
if [[ "${WANDB:-0}" == "1" ]]; then
  WANDB_FLAG="--wandb"
fi

python eval/eval.py \
  --backbone vit \
  --backbone_path "$MODEL" \
  --dataset "$DATASET" \
  --mode "$MODE" \
  --variant "$VARIANT" \
  --text_encoder own \
  --vocab_path "$VOCAB" \
  --lr "$LR" \
  --epochs "$EPOCHS" \
  $WANDB_FLAG \
  "$@"
