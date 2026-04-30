#!/usr/bin/env sh
# NaN 调试启动脚本：默认使用当前可见的全部 GPU，也可用 CUDA_VISIBLE_DEVICES 限制卡数。
# 使用前：conda activate navillm（勿在 POSIX sh 里直接 conda activate）。
#
#   sh scripts/debug_nan.sh
#
# 环境变量可选覆盖：
#   CUDA_VISIBLE_DEVICES=0 sh scripts/debug_nan.sh           # 单卡
#   CUDA_VISIBLE_DEVICES=0,1 sh scripts/debug_nan.sh         # 两卡
#   DEBUG_NPROC_PER_NODE=2 sh scripts/debug_nan.sh           # 显式指定 rank 数
#

set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/build:${PYTHONPATH:-}"

# ---------- 按需修改 ----------
MASTER_PORT="${MASTER_PORT:-41100}"

DATA_DIR="${DATA_DIR:-data}"
LM_PATH="${LM_PATH:-data/models/Vicuna-7B}"
CHECKPOINT="${CHECKPOINT:-/code/NaviLLM/checkpoints/model_with_pretrain.pt}"
# 清空则不加 --resume_from_checkpoint（仅从 Vicuna init）
PRECISION="${PRECISION:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-output/debug_nan}"
# 可选：小规模数据冒烟，例如: EXTRA_ARGS="--max_datapoints 32"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export DEBUG_NAN=1

VISIBLE_GPU_COUNT="$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"
if [ "${VISIBLE_GPU_COUNT}" -le 0 ]; then
  echo "error: no CUDA device is visible. Set CUDA_VISIBLE_DEVICES or check the container GPU allocation." >&2
  exit 1
fi

NPROC_PER_NODE="${DEBUG_NPROC_PER_NODE:-${VISIBLE_GPU_COUNT}}"
case "${NPROC_PER_NODE}" in
  ''|*[!0-9]*)
    echo "error: DEBUG_NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
    exit 1
    ;;
esac
if [ "${NPROC_PER_NODE}" -le 0 ]; then
  echo "error: DEBUG_NPROC_PER_NODE must be > 0, got: ${NPROC_PER_NODE}" >&2
  exit 1
fi
if [ "${VISIBLE_GPU_COUNT}" -lt "${NPROC_PER_NODE}" ]; then
  echo "error: debug_nan.sh would launch ${NPROC_PER_NODE} ranks, but only ${VISIBLE_GPU_COUNT} CUDA device(s) are visible." >&2
  echo "       CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  echo "       Use CUDA_VISIBLE_DEVICES to expose exactly the GPUs you want, or set DEBUG_NPROC_PER_NODE<=${VISIBLE_GPU_COUNT}." >&2
  exit 1
fi

echo "DEBUG_NAN launch: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}, NPROC_PER_NODE=${NPROC_PER_NODE}, visible_gpus=${VISIBLE_GPU_COUNT}"

RESUME_ARGS=""
if [ -n "${CHECKPOINT}" ] && [ -f "${CHECKPOINT}" ]; then
  RESUME_ARGS="--resume_from_checkpoint ${CHECKPOINT}"
elif [ -n "${CHECKPOINT}" ] && [ ! -f "${CHECKPOINT}" ]; then
  echo "warning: CHECKPOINT set but missing: ${CHECKPOINT} (skipped resume)" >&2
fi

# torchrun 续行末尾不要穿插注释。
torchrun --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" train.py \
    --mode train \
    --debug --debug_log_every 20 \
    --stage multi --cfg_file configs/multi.yaml \
    --data_dir "${DATA_DIR}" \
    --pretrained_model_name_or_path "${LM_PATH}" \
    --precision "${PRECISION}" \
    --batch_size 1 \
    --gradient_accumulation_step 1 \
    --num_steps_per_epoch 200 \
    --lr 1e-5 \
    --seed 0 \
    --num_epochs 1 \
    --enable_og --enable_summarize --enable_fgr2r \
    --update_llm true \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj,v_proj \
    --train_datasets R2R \
    --test_datasets R2R \
    --max_saved_checkpoints 0 \
    --output_dir "${OUTPUT_DIR}" \
    ${RESUME_ARGS} \
    ${EXTRA_ARGS}
