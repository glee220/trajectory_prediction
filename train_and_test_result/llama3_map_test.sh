#!/bin/bash 
#SBATCH --job-name=lyj_llama3_map_test  # 定义作业名称
#SBATCH --output=llama3_map_test_%j.log    # 标准输出文件
#SBATCH --cpus-per-task=2    # 为每个任务分配8个CPU核心。
#SBATCH --ntasks=1            # 指定任务的数量为1
#SBATCH --partition=gpujl     # 指定分区为 "gpujl"
#SBATCH --gres=gpu:4         # 请求4个GPU资源
#SBATCH --mem=100GB            # 请求100GB内存（根据需要调整）
#SBATCH --time=10:00:00       # 设置作业运行最大时间为48小时

# 你可以选择排除某个节点
# #SBATCH --exclude=node25

# 1. 加载 Conda 环境
source ~/miniconda3/etc/profile.d/conda.sh     # 根据安装路径调整
conda activate traj_cuda12                     # 切换到 conda 环境

# 随机找一个 > 20000 的空闲端口，避免多任务冲突
master_port=$(( 20000 + RANDOM % 20000 ))
echo "master_port: $master_port"

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

deepspeed  /home/liuyanjiao/trajectory_predict/test_metrics.py \
            --checkpoint_path checkpoints/Exp_llama3__map_train_20250713_165428/best_model.pth \
            --llm_model LLAMA3 \
            --llm_model_path ./models/Meta-Llama-3-8B-Instruct/        \
            --use_map 

# 启动
# sbatch llama3_map_test.sh
# sbatch --mem=0 llama3_map_test.sh
