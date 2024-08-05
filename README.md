# PerceptiveAgent
## Talk With Human-like Agents: Empathetic Dialogue Through Perceptible Acoustic Reception and Reaction (ACL24)

[[Arxiv](https://arxiv.org/abs/2406.12707)] [[Watch Case Study](https://drive.google.com/file/d/1EK1Dqtn5rdDYl306KBg3_ADDN6qhw3v0/view?usp=sharing)]

To avoid overlooking human communication nuances and misinterpreting speakers' intentions, we propose PerceptiveAgent, an empathetic multi-modal dialogue system designed to discern deeper or more subtle meanings beyond the literal interpretations of words through the integration of speech modality perception. Employing LLMs as a cognitive core, PerceptiveAgent perceives acoustic information from input speech and generates empathetic responses based on speaking styles described in natural language. 

![PerceptiveAgent_framework](https://github.com/Haoqiu-Yan/PerceptiveAgent/blob/main/PerceptiveAgent.png)

<!-- [![Watch Case Study](.jpg)](https://drive.google.com/file/d/1EK1Dqtn5rdDYl306KBg3_ADDN6qhw3v0/view?usp=sharing "Watch Case Study") -->

## Getting Started

### Clone this repository.

```bash
git clone https://github.com/Haoqiu-Yan/PerceptiveAgent.git
cd PerceptiveAgent
```

### Configure environment.

Limited by package compatibility, we create two virtual environments. We recommend running on linux using a conda environment. We employ Python 3.8 and Torch 1.13.1 (cuda) in both environments.

1. Environment of Speech Captioner & Chatting with LLM

```bash
conda create -n cap38 python=3.8
conda activate cap38

pip install -r cap_requirement.txt
```

2. Environment of MSMA Synthesizer

```bash
conda create -n syn38 python=3.8
conda activate syn38

pip install -r syn_requirement.txt
```


## Speech Captioner
### Data Preprocessing

For the TextrolSpeeh dataset which provides `caption.csv`, the following command reads captions one by one and saves them as `audio_name.json` files to the output directory `$MIXDIR`. Besides, audio files would be copied to `$MIXDIR` as well.

```bash
DATA_ROOT=/path/datasets/textrolspeech
ALLCSV=${DATA_ROOT}/caption/random_train.csv
MIXDIR=/path/to/save
python ./captioner/dataset/process.py --dataset textrol --data_dir ${DATA_ROOT} \
    --json_path $ALLCSV \
    --saveto $MIXDIR
```

For the other datasets without caption.csv, you can create an empty caption file before processing.
Take the EXPRESSO dataset as an example:

```bash
python data_preprocess/create_caption_expresso.py -audio-root /path/expresso/merge_audio_48khz/ --transcript-path /path/expresso/read_transcriptions.txt --saveas /path/expresso/caption/random_read_all.csv
```
Then, change the argument `--dataset` to the responding dataset.

```bash
DATA_ROOT=/path/expresso
ALLCSV=/path/expresso/caption/random_read_all.csv
MIXDIR=/path/to/save
python ./captioner/dataset/process.py --dataset expresso --data_dir ${DATA_ROOT} \
    --json_path $ALLCSV \
    --saveto $MIXDIR
```


### Training

To update `llama_model` in `configs/capsp_train_gpt2.yaml`, you can download the pretrained Vicuna weights, according to the [instruction](https://github.com/magic-research/bubogpt/blob/main/PrepareVicuna.md) in [BuboGPT](https://github.com/magic-research/bubogpt/blob/main/README.md#models).

To update `ABSOLUTE_PATH_OF_bubogpt_7b` in `configs/capsp_infer.yaml`, you can download the pretrained bubogpt_7b from the [link](https://huggingface.co/magicr/BuboGPT-ckpt/resolve/main/bubogpt_7b.pth).

Then, run the following command to train a speech captioner. 

`bash scripts/captioner_train.sh ${CUDA_ID} ${CUDA_NUM}`

### Inference

Infer by the following shell,

`bash scripts/captioner_infer.sh ${CUDA_ID} ${CUDA_NUM}`


## Chat with LLM

To integrate captions into dialogue history, you can chat with ChatGPT by the following command.

```bash
python ./chat_llm/conversation_chat.py --id-convs ./input_egs/id_convs_eg.txt --audio-asr-caption ./input_egs/audio_asr_caption_eg.txt --save-root /path/to/savedir > ./logs/MMDD-2200pm.log
```

Besides, to send dialogue history only to chatGPT, you can chat by:

```bash
python ./chat_llm/conversation_chat_text-only.py --id-convs ./input_egs//id_convs_eg.txt --audio-asr-caption ./input_egs/audio_asr_caption_eg.txt --save-root /path/to/savedir > ./logs/MMDD-2200pm.log
```


## MSMA Synthesizer

### Data Preprocessing

1. The training phrase takes speech as input. To encode speech into discrete acoustic units, you can run the following command. The HuBert model, specified by the argument `DENSE_NAME` in the shell `expresso_hubert_gen.sh`, would be downloaded automatically.

`bash ./scripts/expresso_hubert_gen.sh`


2. The inference phrase takes text as input. To transform text into discrete acoustic units, a text-to-unit (T2U) model is trained by us, which can be downloaded from [GoogleDrive](). 

The `spm model` trained by us can be download from [GoogleDrive](); Otherwise, you can train your own model by:

`spm_train --input=$ALL_TEXT --model_prefix=spm_bpe_1k --vocab_size=1000 --character_coverage=1.0 --model_type=bpe`

Then, run the following command to transform text into units.

`bash ./scripts/t2u_infer.sh`


### Training

1. Use the [EXPRESSO](https://speechbot.github.io/expresso/), [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) and [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) dataset to pretrain a vocoder conditioned on emotion and speaker labels.
Note: `batch_size` in `$CONFIG_FILE` is the product of batch size and CUDA number.

```bash
CONFIG_FILE=./configs/synthesizer_pretrain_config.json
OUTPUT_DIR=/path/to/save

python -m torch.distributed.launch --nproc_per_node $GPUS --master_port=29502 \
        synthesizer/examples/pretrain/amp_train.py \
        --checkpoint_path $OUTPUT_DIR \
	    --config $CONFIG_FILE \
        --training_epochs 2000 \
        --validation_interval 5000 \
        --checkpoint_interval 25000
```

2. Use the EXPRESSO dataset to finetune the above vocoder conditioned on pitch, energy and speed labels additionally.
To finetune from the lastest checkpoint, you can add `--from-latest-ckpt` in the following command.

```bash
CONFIG_FILE=./configs/synthesizer_finetune_config.json
OUTPUT_DIR=/path/to/save

python -m torch.distributed.launch --nproc_per_node $GPUS \
        synthesizer/examples/mcond_expresso/amp_train.py \
        --checkpoint_path $OUTPUT_DIR \
	    --config $CONFIG_FILE \
        --training_epochs 2000 \
        --validation_interval 5000 \
        --checkpoint_interval 25000 \
        # --from-latest-ckpt
```


### Inference

1. Infer with the pretrained model.

```bash
CUDA_ID=$1
GPUS=$2

INPUT_CODE_FILE=./input_egs/syntheizer_pretrain_val.txt
ckpt=g_00400000
CHECKPOINT_FILE=/path/of/ckptdir/${ckpt}
OUTPUT_DIR=/path/to/savedir
mkdir $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$CUDA_ID python ./synthesizer/examples/pretrain/inference_example.py \
    --input_code_file $INPUT_CODE_FILE \
    --checkpoint_file $CHECKPOINT_FILE \
    --output_dir $OUTPUT_DIR \
    --num-gpu $GPUS
```

2. Infer with the finetuned model.

```bash
CUDA_ID=$1
GPUS=$2

INPUT_CODE_FILE=./input_egs/syntheizer_finetune_dev.txt
ckpt=g_00200000
CHECKPOINT_FILE=/path/of/ckptdir/${ckpt}
OUTPUT_DIR=/path/to/savedir
mkdir $OUTPUT_DIR

CUDA_VISIBLE_DEVICES=$CUDA_ID python ./synthesizer/examples/mcond_expresso/inference_example.py \
    --input_code_file $INPUT_CODE_FILE \
    --checkpoint_file $CHECKPOINT_FILE \
    --output_dir $OUTPUT_DIR \
    --num-gpu $GPUS \
    --dur-prediction 
```

## Author

Haoqiu Yan*, Yongxin Zhu*, Kai Zheng, Bing Liu, Haoyu Cao, Deqiang Jiang and Linli Xu†. (*Equal Contribution, †Corresponding Author)

## How to cite

If you use the code or models from this project in your research, please cite our work as follows:

```Latex
@article{yan2024talk,
  title={Talk With Human-like Agents: Empathetic Dialogue Through Perceptible Acoustic Reception and Reaction},
  author={Yan, Haoqiu and Zhu, Yongxin and Zheng, Kai and Liu, Bing and Cao, Haoyu and Jiang, Deqiang and Xu, Linli},
  journal={arXiv preprint arXiv:2406.12707},
  year={2024}
}
```

## License

PerceptiveAgent is distributed under the Apache License.

## Acknowledgment

This repo is developed based on the following repos:

+ [bubogpt](https://github.com/magic-research/bubogpt)
+ [speech-synthesis](https://github.com/facebookresearch/speech-resynthesis)
+ [fairseq](https://github.com/facebookresearch/fairseq)
+ [textlesslib](https://github.com/facebookresearch/textlesslib)
+ [sentencepiece](https://github.com/google/sentencepiece)
