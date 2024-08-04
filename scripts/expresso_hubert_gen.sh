#!bin/bash
# get discrete codes and datasets, for expresso (read & conversation)

# conda activate fairseq12

set -x
set -e

# 1. manifest, hubert
# for split in {train,dev,test}; do

AUDIO_ROOT=/data1/haoqiuyan/datasets/expresso/merged_audio_16khz/${split}
MANIFEST=./speech_synthesis/hubert_kmn_getcodes/manifest/expresso/manifest_${split}.tsv
TRANSCRIPT=./speech_synthesis/hubert_kmn_getcodes/units/expresso/transcript_${split}

# manifest
# (khz不同, 生成的duration也不同)
python code/synthesis/hubert/my_wav2vec_manifest.py ${AUDIO_ROOT} --desttsv ${MANIFEST} --ext wav --valid-percent 0

# hubert
DENSE_NAME=hubert-base-ls960-layer-9
KMEANS_NAME=kmeans-expresso
VOCAB_SIZE=2000
CUDA_VISIBLE_DEVICES=4 python code/synthesis/hubert/transcribe.py --manifest $MANIFEST  --output $TRANSCRIPT  --dense_model $DENSE_NAME  --quantizer_model $KMEANS_NAME  --vocab_size $VOCAB_SIZE  --preserve_name

done

# 2. generate dataset for vocoder training
for split in {train,dev,test}; do
MANIFEST=./speech_synthesis/hubert_kmn_getcodes/manifest/expresso/manifest_${split}.tsv
TRANSCRIPT=./speech_synthesis/hubert_kmn_getcodes/units/expresso/transcript_${split}
GEN_DATASET=./speech_synthesis/gen_dateset_codes_with_labels/expresso/${split}

python code/synthesis/hubert/gen_datasets_style-only.py --codes ${TRANSCRIPT}.units --manifest $MANIFEST --outdir ${GEN_DATASET} --min-dur 0.01 --tt 0 --cv 0  --spk-prefix expresso
done

# 移动train.txt: dev.txt & test.txt are empty
for split in {train,dev,test}; do
cp ./speech_synthesis/gen_dateset_codes_with_labels/expresso/${split}/train.txt ./speech_synthesis/gen_dateset_codes_with_labels/expresso/${split}.txt
done