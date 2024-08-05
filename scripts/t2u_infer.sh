#!/bin/bash
# text-to-unit infer

set -e
set -x

CUDA_ID=$1

# 1. preprocess
# model=baseline
# model=ours-v2
raw_text=./input_egs/t2u_infer_input.txt
tokenized_text=./t2u/1701_workspace/${model}/dataset/meld.bpe-spm
spm_encode --model=./t2u/spm_bpe_1k.model --output_format=piece < ${raw_text} > ${tokenized_text}

src=bpe-spm
tgt=units

DATA_ROOT=./t2u/1701_workspace/${model}/dataset/data-bin
fairseq-preprocess  --source-lang $src --target-lang $tgt \
  --only-source \
  --srcdict ./t2u/datasets/data-bin/dict.bpe-spm.txt  \
  --trainpref  ./t2u/1701_workspace/${model}/dataset/meld  \
  --destdir ${DATA_ROOT} --workers 4 

cp ./t2u/datasets/data-bin/dict.units.txt $DATA_ROOT

# 2. infer
t2u_model=best_layer4_emb256ffn512_drop1
model_name=${t2u_model}_best
MODEL_PATH=./t2u/models/${t2u_model}/checkpoint_best.pt
TRANSLATION_PATH=./t2u/1701_workspace/${model}/translation

CUDA_VISIBLE_DEVICES=$CUDA_ID fairseq-generate $DATA_ROOT \
    --task=translation --source-lang $src --target-lang $tgt \
    --gen-subset train \
    --path $MODEL_PATH  \
    --max-source-positions 128 --max-target-positions 512 \
    --max-tokens 4096 --sampling --beam 1 --sampling-topk 10 \
    --skip-invalid-size-inputs-valid-test   > ${TRANSLATION_PATH}/${model_name}_tpk10.txt

grep "^D\-" ${TRANSLATION_PATH}/${model_name}_tpk10.txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > ${TRANSLATION_PATH}/${model_name}_tpk10.txt.hypo