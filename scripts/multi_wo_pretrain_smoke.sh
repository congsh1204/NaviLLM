#!/usr/bin/env sh
# 参照 multi_wo_pretrain.sh 的同款环境与小规模训练参数，用于快速验证「训练 + 验证」能否跑通。
# 在 Algorithms/NaviLLM 下执行；请先 conda activate navillm。
#
# 可调环境变量：
#   SMOKE_N=16          每任务最多样本（--max_datapoints）
#   SMOKE_STEPS=8      每个 epoch 的训练步数（--num_steps_per_epoch），远小于正式 2000
#   SMOKE_EPOCHS=1      训练 epoch 数
#   OUTPUT_DIR=output/multi_wo_pretrain_smoke
#   CKPT_PATH=...       若存在则 --resume_from_checkpoint
#   MASTER_PORT=41002   若默认 41000 报 EADDRINUSE，换空闲端口
#   MAX_SAVED_CHECKPOINTS=1   默认 0 不写 epoch_*.pt；设为 1 可在验证提升时保存（与 train.py 逻辑一致）

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/build:${PYTHONPATH}"

export JAVA_HOME="${JAVA_HOME:-$java_path}"
export PATH="$JAVA_HOME/bin:$PATH"
export CLASSPATH=".:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar"

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_COLLNET_ENABLE=0
export NCCL_SOCKET_IFNAME=lo
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# echo "🧹 Cleaning up existing processes..."
# pkill -f torchrun 2>/dev/null
# pkill -f python 2>/dev/null
# sleep 2
MASTER_PORT="${MASTER_PORT:-41000}"

SMOKE_N="${SMOKE_N:-16}"
SMOKE_STEPS="${SMOKE_STEPS:-8}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-1}"
OUT_DIR="${OUTPUT_DIR:-output/multi_wo_pretrain_smoke}"
MAX_CKPT="${MAX_SAVED_CHECKPOINTS:-1}"

torchrun --nnodes=1 --nproc_per_node=4 --master_port "${MASTER_PORT}" train.py \
    --mode train \
    --stage multi --cfg_file configs/multi.yaml \
    --data_dir data --pretrained_model_name_or_path data/models/Vicuna-7B --precision auto \
    --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch "${SMOKE_STEPS}" --lr 3e-5 --seed 0 --num_epochs "${SMOKE_EPOCHS}" \
    --enable_og --enable_summarize --enable_fgr2r \
    --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_target_modules q_proj,v_proj \
    --update_llm true \
    --test_datasets ScanQA CVDN SOON R2R REVERIE \
    --validation_split val_unseen \
    --max_datapoints "${SMOKE_N}" \
    --val_batch_size 2 \
    --max_saved_checkpoints "${MAX_CKPT}" \
    --output_dir "${OUT_DIR}"
