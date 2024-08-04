"""
Code for processing WavCaps dataset.
Usage:
python code/wavcap_process.py --dataset soundbible --data_root /data1/haoqiuyan/describe_speech/prepare/WavCaps/

---
modified by haoqiu
Usage:
python -m dataset.process --dataset textrol --data_dir ./datasets/textrolspeech --json_path ./datasets/textrolspeech/caption/mini_random_train.csv --saveto ./datasets/textrolspeech/random_train/mini
"""


import argparse

import os
from tqdm import tqdm
import glob
import numpy as np
import json

import shutil
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='bbc', choices=['bbc', 'audioset', 'soundbible', 'freesound', 'test', 'textrol', 'expresso', 'meld'])
parser.add_argument(
    "--data_root",
    type=str,
    default="/mnt/bn/zilongdata-hl/dataset/wavcaps"
)
parser.add_argument(
    "--data_dir",
    type=str,
    default="/mnt/bn/zilongdata-hl/dataset/wavcaps/raw_datasets/test/"
)
parser.add_argument(
    "--json_path",
    type=str,
    default="/mnt/bn/zilongdata-hl/dataset/wavcaps/WavCaps/json_files/test.json",
)
parser.add_argument(
    "--saveto",
    type=str,
    default="./datasets/textrolspeech/random_train/mini",
    help="save pairs of audio and json, to a new directionary"
)

parser.add_argument("--split", type=str, help="for meld")
args = parser.parse_args()

DATA_DIRS = {
    "bbc": "raw_datasets/BBC_Sound_Effects_flac/",
    "audioset": "raw_datasets/AudioSet_SL_flac/",
    "soundbible": "raw_datasets/SoundBible_flac/",
    "freesound": "raw_datasets/FreeSound_flac/",
}

JSON_PATHS = {
    "bbc": "WavCaps/json_files/BBC_Sound_Effects/bbc_final.json",
    "audioset": "WavCaps/json_files/AudioSet_SL/as_final.json",
    "soundbible": "WavCaps/json_files/SoundBible/sb_final.json",
    "freesound": "WavCaps/json_files/FreeSound/fsd_final.json",
}



def load_audioset_json(fname):
    """A sample example:
    {   
        'id': 'Yb0RFKhbpFJA.wav',
        'caption': 'Wind and a man speaking are heard, accompanied by buzzing and ticking.',
        'audio': 'wav_path',
        'duration': 10.0
    }
    """
    with open(fname) as f:
        data = json.load(f)

    for sample in data['data']:
        yield sample['id'].split('.')[0], sample['caption'], sample


def load_soundbible_json(fname):
    """A sample example:
    {   
        'title': 'Airplane Landing Airport',
        'description': 'Large commercial airplane landing at an airport runway.',
        'author': 'Daniel Simion',
        'href': '2219-Airplane-Landing-Airport.html',
        'caption': 'An airplane is landing.',
        'id': '2219',
        'duration': 14.1424375,
        'audio': 'wav_path',
        'download_link': 'http://soundbible.com/grab.php?id=2219&type=wav'
    }
    """
    with open(fname) as f:
        data = json.load(f)
    
    for sample in data['data']:
        yield sample['id'], sample['caption'], sample


def load_freesound_json(fname):
    """A sample example:
    {   'id': '180913',
        'file_name': 'UK Mello.wav',
        'href': '/people/Tempouser/sounds/180913/',
        'tags': ['Standard', 'ringtone', 'basic', 'traditional'],
        'description': 'Standard traditional basic ringtone, in mello tone.',
        'author': 'Tempouser',
        'duration': 3.204375,
        'download_link': 'https://freesound.org/people/Tempouser/sounds/180913/download/180913__tempouser__uk-mello.wav',
        'caption': 'A traditional ringtone is playing.',
        'audio': 'wav_path'
    }
    """
    with open(fname) as f:
        data = json.load(f)
    
    for sample in data['data']:
        yield sample['id'], sample['caption'], sample


def load_bbc_json(fname):
    """A sample example:
    {
        'description': "Timber & Wood - Rip saw, carpenters' workshop.",
        'category': "['Machines']",
        'caption': "Someone is using a rip saw in a carpenter's workshop.",
        'id': '07066104',
        'duration': 138.36,
        'audio': 'wav_path',
        'download_link': 'https://sound-effects-media.bbcrewind.co.uk/zip/07066104.wav.zip'
    }
    """
    with open(fname) as f:
        data = json.load(f)
    
    for sample in data['data']:
        yield sample['id'], sample['caption'], sample


def load_test_json(fname):
    """Using SoundBible as a text example."""
    with open(fname) as f:
        data = json.load(f)
    
    for sample in data['data']:
        yield sample['id'], sample['caption'], sample


def load_textrol_json(fname):
    """file type: csv
    columns:
    num,wav_fp,item_name,gender,pitch,tempo,energy,emotion,style_prompt,content_prompt
    ---
    after renamed:
    id,wav_fp,item_name,gender,pitch,tempo,energy,emotion,caption,asr
    """

    df = pd.read_csv(fname)
    df = df.rename(columns={'num':'id', 'style_prompt': 'caption', 'content_prompt': 'asr'})
    for sid, sample in df.iterrows():
        yield sample["wav_fp"], sample["caption"], sample.to_dict()


def load_expresso_json(fname):
    return load_textrol_json(fname)

def load_meld_json(fname):
    return load_textrol_json(fname)



if __name__ == '__main__':
    if not os.path.exists(args.saveto):
        os.makedirs(args.saveto)
        
    if args.dataset in DATA_DIRS:
        data_dir = os.path.join(args.data_root, DATA_DIRS[args.dataset])
        json_path = os.path.join(args.data_root, JSON_PATHS[args.dataset])
    else:
        # process test set or textrolspeech
        data_dir = args.data_dir
        json_path = args.json_path

    if args.dataset == 'expresso':
        print("[INFO]: create json only, without audio copied.")
        
        nofileL = list()
        for audio_path, unsed_caption, meta_data in tqdm(list(globals()[f'load_{args.dataset}_json'](json_path))):
            audio_name = meta_data["item_name"]
            json_save_path = os.path.join(args.saveto,  audio_name.split('.')[0]+ ".json")            
            
            if not os.path.exists(os.path.join(args.saveto,  audio_name)):
                nofileL.append(audio_path)
                continue

            with open(json_save_path, 'w') as f:
                json.dump(meta_data, f)
 
    elif args.dataset == 'meld':
        nofileL = list()
        for audio_path, unsed_caption, meta_data in tqdm(list(globals()[f'load_{args.dataset}_json'](json_path))):
            rename = "{}-{}-{}".format( \
                    meta_data['item_name'].split('.')[0],  \
                    meta_data['gender'].replace('.','').replace('/',''), meta_data['emotion'],
                    )
            # rename = "{}-{}-{}-{}".format( \
            #         meta_data['item_name'].split('.')[0],  \
            #         meta_data['gender'].replace('.',''), meta_data['emotion'],
            #         args.split)
            # argument split is not defined --> change via code
            audio_save_path = os.path.join(args.saveto, rename + "." + audio_path.split('.')[-1])
            json_save_path = os.path.join(args.saveto, rename + ".json")
            
            try:
                # 如果找不到audio, json也不生成了
                shutil.copy(os.path.join(data_dir, audio_path), audio_save_path)
                with open(json_save_path, 'w') as f:
                    json.dump(meta_data, f)
            except FileNotFoundError:
                raise FileNotFoundError(f"json_save_path {json_save_path}\n {os.path.join(data_dir, audio_path)}\n {audio_save_path}")
                nofileL.append(audio_path)
    
    else:
        nofileL = list()
        for audio_path, unsed_caption, meta_data in tqdm(list(globals()[f'load_{args.dataset}_json'](json_path))):
            rename = "{}-{}-{}-{}-{}-{}".format( \
                    meta_data['id'], meta_data['gender'], \
                    meta_data['pitch'], meta_data['tempo'], \
                    meta_data['energy'], meta_data['emotion'])
            audio_save_path = os.path.join(args.saveto, rename + "." + audio_path.split('.')[-1])
            json_save_path = os.path.join(args.saveto, rename + ".json")
            
            try:
                # 如果找不到audio, json也不生成了
                shutil.copy(os.path.join(data_dir, audio_path), audio_save_path)
                with open(json_save_path, 'w') as f:
                    json.dump(meta_data, f)
            except FileNotFoundError:
                nofileL.append(audio_path)
        
    
    if nofileL:
        print("Audio not found: {}".format(nofileL))


    # file_list = glob.glob(f'{data_dir}/*.flac')
    # 根据数据集选择相应的func处理json
    # for data_id, unsed_caption, meta_data in tqdm(list(globals()[f'load_{args.dataset}_json'](json_path))):
        # file_name = os.path.join(data_dir, data_id + '.flac')
        # json_save_path = os.path.join(data_dir, data_id + '.json')
        # text_save_path = os.path.join(data_dir, data_id + '.text')
        # file_list.remove(file_name)
        

        # assert os.path.exists(file_name), f'{file_name} does not exist!'
        # with open(json_save_path, 'w') as f:
        #     json.dump(meta_data, f)
    
    # 删除没有caption的audio
    # if len(file_list) > 0:
    #     # import pdb; pdb.set_trace()
    #     for f in file_list:
    #         os.remove(f)
  

    # file_list = glob.glob(f'{data_dir}/*.flac')
    # for file_path in file_list:
    #     audio_json_save_path = file_path.replace('.flac', '.json')
    #     audio_text_save_path = file_path.replace('.flac', '.text')
