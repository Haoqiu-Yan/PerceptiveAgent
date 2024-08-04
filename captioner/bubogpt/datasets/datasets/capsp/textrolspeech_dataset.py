"""
ref:
+ /data/haoqiuyan/github_repo/LAVIS/lavis/datasets/builders/caption_builder.py
+ /data/haoqiuyan/describe_speech/code/capsp/bubogpt/datasets/datasets/audio_caption/audio_caption_datasets.py
---
author: haoqiuyan
"""


import os
from glob import glob
import json
import torchaudio

from torch.utils.data import Dataset, default_collate
import webdataset as wds
# from bubogpt.datasets.datasets.base_dataset import BaseDualDataset


class BaseAudioDataset(Dataset):
    def __init__(
        self, audio_processor=None, text_processor=None, location=None
    ):
        """
        location: directory of json and wav
                (eg.) datasets/textrolspeech/random_train/mini
        """
        self.data_root = location

        self.annotation = []
        ann_paths = glob(os.path.join(location, "*.json"))
        for ann_path in ann_paths:
            ann = json.load(open(ann_path, "r"))
            ann["audio_share_name"] = ann_path.split(".")[0]
            # self.annotation.extend(ann)
            self.annotation.append(ann)

        self.audio_processor = audio_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, audio_processor, text_processor):
        self.audio_processor = audio_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class CapspDataset(BaseAudioDataset):
    def __init__(self, audio_processor, text_processor, location):
        super().__init__(audio_processor=audio_processor, text_processor=text_processor, location=location)

        
    def load_audio(self, fname):
        
        return torchaudio.load(fname)

    def __getitem__(self, index):
        extension = ".wav"
        ann = self.annotation[index]

        audio_path = ann["audio_share_name"] + extension
        audio = self.load_audio(audio_path)
        audio = self.audio_processor(audio)

        caption = self.text_processor(ann["caption"])

        return {
            "audio": audio,
            "text_input": caption,
            # "image_id": self.img_ids[ann["image_id"]],
        }
        
        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode(wds.torch_audio, handler=wds.warn_and_continue),
            wds.to_tuple("wav", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.x_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    # def to_dict(self, sample):
    #     return {
    #         "audio": sample[0],
    #         # [clips_per_video, channel, mel_bins, time_steps]
    #         "text_input": self.text_processor(sample[1]["caption"]),
    #     }


class CapspEvalDataset(CapspDataset):
    """save audio ids"""

    
    def __init__(self, audio_processor, text_processor, location):
        super().__init__(audio_processor, text_processor, location)


    # def to_dict(self, sample):
    #     return {
    #         "id": sample[1]["wav_fp"],
    #         "audio": sample[0],
    #         # [clips_per_video, channel, mel_bins, time_steps]
    #         "text_input": self.text_processor(sample[1]["caption"]),
    #     }
    
    def __getitem__(self, index):
        extension = ".wav"
        ann = self.annotation[index]

        audio_path = ann["audio_share_name"] + extension
        audio = self.load_audio(audio_path)
        audio = self.audio_processor(audio)

        caption = self.text_processor(ann["caption"])

        return {
            "id": ann["wav_fp"],
            "audio": audio,
            "text_input": caption,
            }