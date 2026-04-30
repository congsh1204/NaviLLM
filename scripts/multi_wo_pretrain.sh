#!/usr/bin/env sh

# set mp3d path (resolve repo root from this script location)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/build:${PYTHONPATH}"

# set java path
export JAVA_HOME=$java_path
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# conda activate 在 `sh scripts/...` 下不可用；请先: conda activate navillm
# 或在 bash 里: source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate navillm


export NCCL_SHM_DISABLE=1
export NCCL_NVLS_ENABLE=0
export NCCL_COLLNET_ENABLE=0
export NCCL_SOCKET_IFNAME=lo
export TORCH_DISTRIBUTED_DEBUG=OFF

# ---------- torchrun 参数说明（勿在续行 \ 之间插入 # 注释，否则会打断命令）----------
# --nnodes=1: 单机训练
# --nproc_per_node=4: 单机 4 进程（通常 4 卡）
# --master_port: 分布式端口（可用环境变量 MASTER_PORT 覆盖）
# --stage multi / --cfg_file: 多任务阶段与配置
# --data_dir / --pretrained_model_name_or_path: 数据根目录与 Vicuna 路径
# --precision: 混合精度
# --batch_size / --gradient_accumulation_step / --num_steps_per_epoch / --lr / --seed / --num_epochs
# --enable_og / --enable_summarize / --enable_fgr2r: 任务开关
# --use_lora 及 lora_*: LoRA 微调（需 pip install peft）
# --test_datasets: 验证集
# --max_saved_checkpoints / --output_dir: checkpoint 与输出目录
# 精度映射（--precision auto）：
# V100 -> fp16
# A100 -> amp_bf16

# training for 30 epochs
torchrun --nnodes=1 --nproc_per_node=4 --master_port 41000 train.py \
    --mode train \
    --stage multi --cfg_file configs/multi.yaml \
    --data_dir data --pretrained_model_name_or_path data/models/Vicuna-7B --precision auto \
    --batch_size 1 --gradient_accumulation_step 8 --num_steps_per_epoch 2000 --lr 1e-5 --seed 0 --num_epochs 30 \
    --enable_og --enable_summarize --enable_fgr2r \
    --update_llm true --use_lora --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_target_modules q_proj,v_proj \
    --train_datasets R2R \
    --test_datasets R2R \
    --max_saved_checkpoints 1 --output_dir output/multi_wo_pretrain \
    --resume_from_checkpoint /code/NaviLLM/checkpoints/model_with_pretrain.pt \
    