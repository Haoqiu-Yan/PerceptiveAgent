"""
Create caption file like textrolspeech (csv).
for expresso
------
Usage:
python code/expresso/create_caption_csv.py --audio-root [file] --transcript-path [file] --saveas [file]

python code/expresso/create_caption_csv.py --audio-root ./datasets/expresso/merge_audio_48khz/ --transcript-path ./datasets/expresso/read_transcriptions.txt --saveas ./datasets/expresso/caption/random_read_all_tmp.csv

"""


import os
import json
import argparse
from pathlib import Path
from glob import glob

import pandas as pd


def get_attributes_from_transcript(file_path):
    textDF = pd.read_csv(file_path, delimiter='\t', names=['item_name', 'content_prompt'])
    # textDF.index.name = 'id'
    textDF['item_name'] = textDF.item_name.map(lambda x: x + '.wav')

    return textDF


def get_attributes_from_audio(audio_root):
    attributes = ['wav_fp','item_name','gender','pitch','tempo','energy','emotion','style_prompt',]
    attributesD = {a:[] for a in attributes}

    audio_paths = glob(os.path.join(audio_root, "*.wav"))
    for audio_path in audio_paths:    
        audio_name = audio_path.split('/')[-1]
        try:
            speaker, emotion, aid = audio_name.split('.')[0].split('_')
        except ValueError:
            # some audios have type {emphasis, longform}
            speaker, emotion, type, aid = audio_name.split('.')[0].split('_')

        attributesD['wav_fp'].append(audio_path)
        attributesD['item_name'].append(audio_name)
        attributesD['gender'].append('F' if (int(speaker[-1]) % 2) == 0 else 'M')
        attributesD['emotion'].append(emotion)

        attributesD['pitch'].append('UNK')
        attributesD['tempo'].append('UNK')
        attributesD['energy'].append('UNK')
        attributesD['style_prompt'].append('UNK')
        # attributesD['content_prompt'].append('UNK')
    
    attributesDF = pd.DataFrame(attributesD)
    # attributesDF.index.name = 'id'

    return attributesDF


def get_attributes_merged(attributesDF, textDF, save_path):
    mergedDF = textDF.set_index('item_name').join(attributesDF.set_index('item_name'))
    mergedDF = mergedDF.reset_index()
    mergedDF.index.name = 'num'

    mergedDF.to_csv(save_path)
    print(f"Writing to {save_path}")

    return mergedDF


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-root", required=True, type=Path, help="audio root directory"
    )
    parser.add_argument(
        "--transcript-path", required=True, type=Path, help="transcript file path"
    )
    parser.add_argument(
        "--saveas", required=True, type=Path, help="save caption path"
    )
    

    args = parser.parse_args()

    audio_df = get_attributes_from_audio(args.audio_root)
    text_df = get_attributes_from_transcript(args.transcript_path)
    get_attributes_merged(attributesDF=audio_df, textDF=text_df, save_path=args.saveas)


if __name__ == '__main__':
    main()
