#!/bin/bash
# modified from ~/github_repo/bubogpt/dist_train.sh
# bash scripts/captioner_train.sh 0,1,2,3 4 2>&1 | tee ./logs/1109-2010pm.log
set -x

CUDA_ID=$1
GPUS=$2

CODE_ROOT=./captioner

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# GPUS=${GPUS:-${ARNOLD_WORKER_GPU}}

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.2"}
MASTER_PORT=${MASTER_PORT:-9902}

# settings for torch log
export BYTED_TORCH_FX=O1
export BYTED_TORCH_BYTECCL=O1
export TOKENIZERS_PARALLELISM=false
export HADOOP_ROOT_LOGGER=error,console

# settings for DDP multi-node for lab.pytorch image >= 1.13
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

export NCCL_SOCKET_IFNAME=eno1
export NCCL_SHM_DISABLE=0

# add
# export TORCH_CPP_LOG_LEVEL=INFO NCCL_DEBUG=INFO
# NCCL_IB_TIMEOUT=22
# NCCL_IB_TC=128

# start training
CONFIG=./configs/capsp_train_gpt2.yaml
CUDA_VISIBLE_DEVICES=$CUDA_ID torchrun --nnodes=$NNODES \
         --node_rank=$NODE_RANK \
         --nproc_per_node=$GPUS \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         ${CODE_ROOT}/train.py \
         --cfg-path \
         $CONFIG \
         ${@:3}
