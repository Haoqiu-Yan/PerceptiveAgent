#!/bin/bash
# for inference
# bash scripts/captioner_infer.sh 0,1,2,3 4

set -x

CODE_ROOT=./captioner
CONFIG=./configs/capsp_infer.yaml

CUDA_ID=$1
GPUS=$2

CUDA_VISIBLE_DEVICES=$CUDA_ID python -m torch.distributed.run --nproc_per_node=$GPUS \
        ${CODE_ROOT}/evaluate_capsp.py \
        --cfg-path ${CONFIG}
