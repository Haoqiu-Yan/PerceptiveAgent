"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from bubogpt.processors.base_processor import BaseProcessor
from bubogpt.processors.blip_processors import (
    Blip2ImageTrainProcessor,
    Blip2ImageEvalProcessor,
    BlipCaptionProcessor,
)
from bubogpt.processors.imagebind_vision_processor import (
    ImageBindCaptionProcessor,
    ImageBindVisionTrainProcessor,
    ImageBindVisionEvalProcessor
)
from bubogpt.processors.imagebind_audio_processor import (
    ImageBindAudioTrainProcessor,
    ImageBindAudioEvalProcessor,
)

from bubogpt.processors.wav2vec_audio_processor import (
    Wav2vecAudioTrainProcessor,
    Wav2vecAudioEvalProcessor,
)

from bubogpt.common.registry import registry

__all__ = [
    "BaseProcessor",
    "Blip2ImageTrainProcessor",
    "Blip2ImageEvalProcessor",
    "BlipCaptionProcessor",
    "ImageBindCaptionProcessor",
    "ImageBindVisionTrainProcessor",
    "ImageBindVisionEvalProcessor",
    "ImageBindAudioTrainProcessor",
    "ImageBindAudioEvalProcessor",
    "Wav2vecAudioTrainProcessor",
    "Wav2vecAudioEvalProcessor",
]


def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
