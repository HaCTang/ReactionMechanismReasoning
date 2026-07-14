#!/bin/bash
# Paper SFT recipe (SFT HQ): Qwen3-30B-A3B-Instruct on johnny-w/flower:sft_hq (3,650).
# Requires 8x GPUs with >=80GB each for FSDP.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${PROJECT_DIR}/.venv/bin/activate" 2>/dev/null || true

export HF_TOKEN="${HF_TOKEN:-$(cat "${PROJECT_DIR}/hf_token.key" 2>/dev/null || true)}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen3-reaction-sft}"
export WANDB_NAME="${WANDB_NAME:-qwen3-30b-pathway-hq}"

MODEL="${MODEL:-${PROJECT_DIR}/checkpoints/Qwen3-30B-A3B-Instruct}"
DATA="${DATA:-johnny-w/flower:sft_hq}"
OUT="${OUT:-${PROJECT_DIR}/checkpoints/qwen3-30b-pathway-hq}"
NPROC="${NPROC:-8}"

echo "SFT HQ | model=${MODEL} | data=${DATA} | out=${OUT}"

torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT:-29500}" \
  "${PROJECT_DIR}/train/train.py" \
  --dataset_type pathway \
  --model_name "${MODEL}" \
  --data_path "${DATA}" \
  --output_dir "${OUT}" \
  --num_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --warmup_steps 100 \
  --lr_scheduler_type cosine \
  --max_length 16384 \
  --logging_steps 10 \
  --save_total_limit 1 \
  --save_strategy epoch \
  --eval_steps 500 \
  --use_fsdp_activation_checkpointing \
  --report_to "${REPORT_TO:-wandb}"
