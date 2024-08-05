#!bin/bash
# get discrete codes and datasets, for expresso (read & conversation)

# conda activate fairseq12

set -x
set -e

# 1. get acoustic units encoded by a hubert model

AUDIO_ROOT=/path/to/audiodir
MANIFEST=/path/of/generated/manifest
TRANSCRIPT=/path/of/generated/units

# manifest
# (khz不同, 生成的duration也不同)
python ./synthesizer/hubert/modified_wav2vec_manifest.py ${AUDIO_ROOT} --desttsv ${MANIFEST} --ext wav --valid-percent 0

# hubert
DENSE_NAME=hubert-base-ls960-layer-9
KMEANS_NAME=kmeans-expresso
VOCAB_SIZE=2000
CUDA_VISIBLE_DEVICES=4 python ./synthesizer/hubert/transcribe.py --manifest $MANIFEST  --output $TRANSCRIPT  --dense_model $DENSE_NAME  --quantizer_model $KMEANS_NAME  --vocab_size $VOCAB_SIZE  --preserve_name


# 2. construct dataset for vocoder training
GEN_DATASET=/path/to/saveas

python ./synthesizer/hubert/gen_datasets_style-only.py --codes ${TRANSCRIPT}.units --manifest $MANIFEST --outdir ${GEN_DATASET} --min-dur 0.01 --tt 0 --cv 0  --spk-prefix expresso

# Note:dev.txt & test.txt are empty
