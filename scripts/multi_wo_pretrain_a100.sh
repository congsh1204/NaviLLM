#!/usr/bin/env sh

export PYTHONPATH=/root/mount/Matterport3DSimulator/build:$PYTHONPATH

# set java path
export JAVA_HOME=$java_path
export PATH=$JAVA_HOME/bin:$PATH
export CLASSPATH=.:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar

# conda activate 在 `sh scripts/...` 下不可用；请先: conda activate navillm
# 或在 bash 里: source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate navillm


# NCCL/torchrun settings for single-node Docker training.
# 2026-05-05 base: 单机 Docker 训练，禁用 IB/RDMA 探测，并固定 bootstrap 走 lo。
#   - NCCL_IB_DISABLE=1: 避免无 InfiniBand 环境下探测 RDMA/libibverbs。
#   - NCCL_SOCKET_IFNAME=lo: 单机进程间初始化走本机 loopback；多机训练不能用 lo。
# 2026-05-05 trial-1: 仅新增 NCCL_CUMEM_ENABLE=0，保留 GPU P2P，尝试规避 P2P/CUMEM ALLGATHER 超时且尽量保留速度。
#   - 结果: 仍然 ALLGATHER 超时。
# 2026-05-05 trial-2: 新增 NCCL_P2P_LEVEL=PXB，限制 P2P 只走较近 PCIe 路径，保留部分 P2P 性能。
#   - 若仍超时，下一步使用 NCCL_P2P_DISABLE=1 完全禁用 P2P。
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export NCCL_CUMEM_ENABLE=0
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=OFF # 关闭分布式详细日志；排查时可临时改为 DETAIL

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
torchrun --nnodes=1 --nproc_per_node=1 --master_port 41000 train.py \
    --mode train \
    --stage multi --cfg_file configs/multi.yaml \
    --data_dir data --pretrained_model_name_or_path data/models/Vicuna-7B --precision auto \
    --batch_size 1 --gradient_accumulation_step 16 --num_steps_per_epoch 2000 --lr 3e-5 --seed 0 --num_epochs 60 \
    --enable_og --enable_summarize --enable_fgr2r \
    --update_llm true --use_lora --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --lora_target_modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --test_datasets R2R \
    --max_saved_checkpoints 1 --output_dir output/multi_wo_pretrain_a100_5.5 
    