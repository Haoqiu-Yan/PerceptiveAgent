#!/bin/bash
# text-to-unit infer

set -e
set -x

CUDA_ID=$1

# 1. preprocess
raw_text=./input_egs/t2u_infer_input.txt
tokenized_text=/path/to/save/t2u_infer_input.bpe-spm
# Train a spm model on your own dataset.
# spm_train --input=$ALL_TEXT --model_prefix=spm_bpe_1k --vocab_size=1000 --character_coverage=1.0 --model_type=bpe
spm_encode --model=/path/spm_bpe_1k.model --output_format=piece < ${raw_text} > ${tokenized_text}

src=bpe-spm
tgt=units

DATA_ROOT=/path/to/save/data-bin
fairseq-preprocess  --source-lang $src --target-lang $tgt \
  --only-source \
  --srcdict ./input_egs/dict.bpe-spm.txt  \
  --trainpref  ./output_t2u/t2u_infer_input  \
  --destdir ${DATA_ROOT} --workers 4 

cp ./input_egs/dict.units.txt $DATA_ROOT

# 2. infer
MODEL_PATH=/path/t2u_checkpoint.pt
TRANSLATION_PATH=/path/to/savedir/translation

CUDA_VISIBLE_DEVICES=$CUDA_ID fairseq-generate $DATA_ROOT \
    --task=translation --source-lang $src --target-lang $tgt \
    --gen-subset train \
    --path $MODEL_PATH  \
    --max-source-positions 128 --max-target-positions 512 \
    --max-tokens 4096 --sampling --beam 1 --sampling-topk 10 \
    --skip-invalid-size-inputs-valid-test   > ${TRANSLATION_PATH}/output_tpk10.txt

grep "^D\-" ${TRANSLATION_PATH}/output_tpk10.txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > ${TRANSLATION_PATH}/output_tpk10.hypo