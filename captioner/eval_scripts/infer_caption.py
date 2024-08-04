import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from constants.constant import LIGHTER_COLOR_MAP_HEX
# NOTE: Must import LlamaTokenizer before `bubogpt.common.config`
# otherwise, it will cause seg fault when `llama_tokenizer.decode` is called

from grounding_model import GroundingModule
from match import MatchModule
from bubogpt.common.config import Config
from bubogpt.common.dist_utils import get_rank
from bubogpt.common.registry import registry
from eval_scripts.conversation import Chat, CONV_X, DummyChat
# NOTE&TODO: put this before bubogpt import will cause circular import
# possibly because `imagebind` imports `bubogpt` and `bubogpt` also imports `imagebind`
from imagebind.models.image_bind import ModalityType
# from ner import NERModule
from tagging_model import TaggingModule


class InferCaption():
    def __init__(self, cfg:Config):
        self.datasets_cfg = cfg.datasets_cfg
        self.model_cfg = cfg.model_cfg
        self.run_cfg = cfg.run_cfg
        
        self.processors = self._get_processor_fromcfg()
        self.model = self._get_model_fromcfg()
    
        

    def _get_processor_fromcfg(self):
        # only 1 item is contained in datasets_cfg
        for name in self.datasets_cfg:
            aud_processor_cfg = self.datasets_cfg[name].audio_processor.eval
            aud_processor = registry.get_processor_class(aud_processor_cfg.name).from_config(aud_processor_cfg)
        processors = {ModalityType.AUDIO: aud_processor}

        return processors

    def _get_model_fromcfg(self):
        """
        load capsp model from config with chkpt
        ---
        return: model: class CAPSP(BaseModel)
        """
        model_cls = registry.get_model_class(self.model_config.arch)
        model = model_cls.from_config(self.model_config).cuda()

        return model


    def load_audio(self):
        """preprocess audio: input, type of tar"""
        


    def load_dataset(self):
        pass

    def model(self):
        pass