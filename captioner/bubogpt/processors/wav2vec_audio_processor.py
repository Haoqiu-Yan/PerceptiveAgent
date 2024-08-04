"""
write for capsp
---
Author: haoqiu
"""

from bubogpt.common.registry import registry
from bubogpt.processors.base_processor import BaseProcessor

import torch
import torchaudio
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn import ZeroPad2d

import logging


class Wav2vecAudioBaseProcessor(BaseProcessor):
    def __init__(self, target_sr=None, max_duration=None,):
        self.target_sr = 16000 if target_sr is None else target_sr
        self.max_duration = max_duration

    def __call__(self, item):
        # item: Tuple[wavform tensor, sampling rate int]
        waveform, origin_sr = item[0], item[1]
        waveform = self.waveform_resample(waveform, origin_sr)
        try:
            waveform = waveform.reshape(waveform.shape[1])
        except RuntimeError:
            # 对MEAD dataset, 2 channel -> 1 channel
            waveform = waveform[0].reshape(waveform.shape[1])

        max_wav_len = self.max_duration * self.target_sr
        if waveform.shape[0] < max_wav_len:
            right_pad = torch.zeros((max_wav_len - waveform.shape[0]))
            waveform = torch.cat([waveform, right_pad])
        elif waveform.shape[0] > max_wav_len:
            logging.info(f"##CUT##: sample lens is {waveform.shape[0]}, larger than {max_wav_len}")
            waveform = waveform[:max_wav_len]

        return waveform

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        target_sr = cfg.get("target_sr", 16000)
        max_duration = cfg.get("max_duration", 500000)
        

        return cls(
            target_sr=target_sr,
            max_duration=max_duration,
        )

    def waveform_resample(self, waveform: Tensor, origin_sr: int) -> Tensor:
        waveform = torchaudio.functional.resample(waveform, orig_freq=origin_sr, new_freq=self.target_sr)
        duration = waveform.size(1) / self.target_sr
        # TODO: if audio is too long, then cut its tail.
        # if duration > self.max_duration:

        return waveform


@registry.register_processor("wav2vec_audio_train")
class Wav2vecAudioTrainProcessor(Wav2vecAudioBaseProcessor):
    def __init__(self, target_sr=None, max_duration=None,):
        super().__init__(target_sr=target_sr, max_duration=max_duration,)


@registry.register_processor("wav2vec_audio_eval")
class Wav2vecAudioEvalProcessor(Wav2vecAudioBaseProcessor):
    def __init__(self, target_sr=None, max_duration=None,):
        super().__init__(target_sr=target_sr, max_duration=max_duration,)
