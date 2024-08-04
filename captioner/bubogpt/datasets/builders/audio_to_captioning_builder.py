import os
import logging
import warnings

from bubogpt.common.registry import registry
from bubogpt.datasets.builders.capsp_base_dataset_builder import CapspBaseDatasetBuilder
from bubogpt.datasets.datasets.capsp import CapspDataset, CapspEvalDataset


@registry.register_builder("textrolspeech")
class TextrolSpeechBuilder(CapspBaseDatasetBuilder):
    """ 1. take wav as audio input
        2. train: 
          eval: save audio_id, no infinite dataloader (for val & test) """

    DATASET_CONFIG_DICT = {"default": "configs/datasets/textrolspeech/defaults.yaml", \
                           "train": "configs/datasets/textrolspeech/train.yaml", \
                            "infer_mini": "configs/datasets/textrolspeech/infer_mini.yaml"}
    
    train_dataset_cls = CapspDataset
    eval_dataset_cls = CapspEvalDataset