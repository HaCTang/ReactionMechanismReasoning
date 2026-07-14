#!/bin/bash
# Paper GSPO recipe (v6): SFT HQ -> GSPO on johnny-w/flower:rl_v6 (2,000 samples).
# Reported result: FukuyamaBench Set A exact pass@1 ~8.3% (v6 ep3).
# Requires verl (see README) and 8x GPUs.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "${PROJECT_DIR}/.venv/bin/activate" 2>/dev/null || true

export HF_TOKEN="${HF_TOKEN:-$(cat "${PROJECT_DIR}/hf_token.key" 2>/dev/null || true)}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-1}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"

MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}/checkpoints/qwen3-30b-pathway-hq}"
TRAIN_DATA="${TRAIN_DATA:-${PROJECT_DIR}/data/verl_v6/train.parquet}"
VAL_DATA="${VAL_DATA:-${PROJECT_DIR}/data/verl_v6/validation.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/checkpoints/qwen3-30b-gspo-v6}"

LR=1e-6
TOTAL_EPOCHS=3
N_RESP=16
CLIP_LOW=3e-4
CLIP_HIGH=4e-4
TRAIN_BSZ=16
MINI_BSZ=4
MAX_PROMPT=2048
MAX_RESP=12288
ACTOR_MAX_TOKEN=$(( (MAX_PROMPT + MAX_RESP) * 2 ))
SAVE_FREQ=125
TEST_FREQ=62

echo "GSPO v6 | model=${MODEL_PATH} | data=${TRAIN_DATA} | out=${OUTPUT_DIR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="${TRAIN_DATA}" \
  data.val_files="${VAL_DATA}" \
  data.return_raw_chat=True \
  data.train_batch_size=${TRAIN_BSZ} \
  data.max_prompt_length=${MAX_PROMPT} \
  data.max_response_length=${MAX_RESP} \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=8 \
  data.truncation=error \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=${LR} \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.optim.clip_grad=1.0 \
  actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.clip_ratio_low=${CLIP_LOW} \
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_HIGH} \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKEN} \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.checkpoint.save_contents='["model","extra"]' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=${N_RESP} \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  reward.reward_manager.name=naive \
  ++reward.custom_reward_function.path="${PROJECT_DIR}/rl/chemistry_rewards.py" \
  ++reward.custom_reward_function.name=compute_score \
  trainer.total_epochs=${TOTAL_EPOCHS} \
  trainer.save_freq=${SAVE_FREQ} \
  trainer.test_freq=${TEST_FREQ} \
  trainer.logger='["console","wandb"]' \
  trainer.project_name=qwen3-reaction-rl \
  trainer.experiment_name=gspo-v6-qwen3-30b \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.val_before_train=False
