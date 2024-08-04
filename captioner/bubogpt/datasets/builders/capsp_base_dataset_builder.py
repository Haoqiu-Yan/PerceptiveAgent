"""
ref: 
+ /data/haoqiuyan/github_repo/LAVIS/lavis/datasets/builders/base_dataset_builder.py
+ /data/haoqiuyan/describe_speech/code/capsp/bubogpt/datasets/builders/audio_text_pair_builder.py
---
author: haoqiu
"""


import logging
import os
import shutil
import warnings

from omegaconf import OmegaConf
import torch.distributed as dist
from torchvision.datasets.utils import download_url

import bubogpt.common.utils as utils
from bubogpt.common.dist_utils import is_dist_avail_and_initialized, is_main_process
from bubogpt.common.registry import registry
from bubogpt.datasets.builders import load_dataset_config
from bubogpt.processors.base_processor import BaseProcessor


class CapspBaseDatasetBuilder:
    train_dataset_cls, eval_dataset_cls = None, None

    def __init__(self, cfg=None):
        super().__init__()

        if cfg is None:
            # help to create datasets from default config.
            self.config = load_dataset_config(self.default_config_path())
        elif isinstance(cfg, str):
            self.config = load_dataset_config(cfg)
        else:
            # when called from task.build_dataset()
            self.config = cfg

        self.data_type = self.config.data_type

        self.audio_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}
        self.text_processors = {"train": BaseProcessor(), "eval": BaseProcessor()}

    def build_datasets(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed

        # if is_main_process():
        #     self._download_data()

        if is_dist_avail_and_initialized():
            dist.barrier()

        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        datasets = self.build()  # dataset['train'/'val'/'test']

        return datasets

    def build_processors(self):
        aud_proc_cfg = self.config.get("audio_processor")
        txt_proc_cfg = self.config.get("text_processor")

        if aud_proc_cfg is not None:
            aud_train_cfg = aud_proc_cfg.get("train")
            aud_eval_cfg = aud_proc_cfg.get("eval")

            self.audio_processors["train"] = self._build_proc_from_cfg(aud_train_cfg)
            # modified by haoqiu (改回来了)
            self.audio_processors["eval"] = self._build_proc_from_cfg(aud_eval_cfg)
            # self.audio_processors["test"] = self._build_proc_from_cfg(aud_eval_cfg)
            # -----------

        if txt_proc_cfg is not None:
            txt_train_cfg = txt_proc_cfg.get("train")
            txt_eval_cfg = txt_proc_cfg.get("eval")

            self.text_processors["train"] = self._build_proc_from_cfg(txt_train_cfg)
            # modified by haoqiu
            self.text_processors["eval"] = self._build_proc_from_cfg(txt_eval_cfg)
            # self.text_processors["test"] = self._build_proc_from_cfg(txt_eval_cfg)
            # -----------

    @staticmethod
    def _build_proc_from_cfg(cfg):
        return (
            registry.get_processor_class(cfg.name).from_config(cfg)
            if cfg is not None
            else None
        )

    @classmethod
    def default_config_path(cls, type="default"):
        return utils.get_abs_path(cls.DATASET_CONFIG_DICT[type])

    def _download_data(self):
        self._download_ann()
        self._download_aud()

    def _download_ann(self):
        raise NotImplementedError
    
    def _download_aud(self):
        raise NotImplementedError

    def build(self):
        """
        train: 
        eval: save audio_id, no infinite dataloader (for val & test)
        """

        self.build_processors()

        build_info = self.config.build_info

        # ann_info = build_info.annotations
        # vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in build_info.keys():
            if split not in ["train", "val", "test"]:
                continue

            is_train = split == "train"

            # processors
            audio_processor = (
                self.audio_processors["train"]
                if is_train
                else self.audio_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"]
                if is_train
                else self.text_processors["eval"]
            )

            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls

            # if dataset is websetdatatset class
            # datasets[split] = dataset_cls(
            #     audio_processor=audio_processor,
            #     text_processor=text_processor,
            #     location=build_info[split].storage,
            # ).inner_dataset
            
            # if dataset is acquired one by one
            datasets[split] = dataset_cls(
                audio_processor=audio_processor,
                text_processor=text_processor,
                location=build_info[split].storage,
            )

        return datasets