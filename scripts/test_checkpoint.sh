#!/usr/bin/env sh

# Matterport PYTHONPATH（与 multi_wo_pretrain.sh 相同）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/build:${PYTHONPATH}"

# ---------- 评测参数：按需改下面这些变量 ----------
CHECKPOINT=checkpoints/model_without_pretrain.pt
NPROC_PER_NODE=4
MASTER_PORT=41005
DATA_DIR=data
LM_PATH=data/models/Vicuna-7B
PRECISION=amp_bf16
VALIDATION_SPLIT=val_unseen
OUTPUT_DIR=output/eval_checkpoint
VAL_BATCH_SIZE=2

# v100 可改为 PRECISION=fp16 或 fp32；与训练该权重时保持一致。
# 省略 --update_llm 时默认 false（冻结 LM）。LoRA 权重须加 --use_lora … 且必须 --update_llm true。
# ------------------------------------------------------------------

torchrun --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --master_port "${MASTER_PORT}" train.py \
    --mode test \
    --stage multi --cfg_file configs/multi.yaml \
    --data_dir "${DATA_DIR}" --pretrained_model_name_or_path "${LM_PATH}" --precision "${PRECISION}" \
    --batch_size 1 --val_batch_size "${VAL_BATCH_SIZE}" \
    --enable_og --enable_summarize --enable_fgr2r \
    --test_datasets R2R \
    --validation_split "${VALIDATION_SPLIT}" \
    --max_saved_checkpoints 0 --output_dir "${OUTPUT_DIR}" \
    --resume_from_checkpoint "${CHECKPOINT}"
