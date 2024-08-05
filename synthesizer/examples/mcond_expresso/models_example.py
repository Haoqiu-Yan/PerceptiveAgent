# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import torch
import torch.nn as nn
import torch.nn.functional as F

from examples.speech_to_speech_translation.models import (
    DurationCodeGenerator,
    process_duration,
)


class MultiConditionsCodeGenerator(DurationCodeGenerator):
    """
    Discrete unit-based HiFi-GAN vocoder with one-hot expression conditioning.
    (used in the Expresso paper)
    The current implementation only supports unit, speaker ID and
    expression ID input and does not support F0 input.
    It also support duration prediction as in examples/speech_to_speech_translation.
    """

    def __init__(self, h):
        super().__init__(h)
        self.multistyle = h.get("multistyle", None)
        self.multiconds = h.get("multiconds", None)

        if self.multispkr:
            self.spkr = nn.Embedding(h.get("num_speakers", 200), h.embedding_dim)
        if self.multistyle:
            self.style = nn.Embedding(h.get("num_styles", 100), h.embedding_dim)
        if self.multiconds:
            self.pitch = nn.Embedding(h.get("num_pitchs", 3), h.embedding_dim)
            self.speed = nn.Embedding(h.get("num_speeds", 3), h.embedding_dim)
            self.energy = nn.Embedding(h.get("num_energys", 3), h.embedding_dim)
            self.project_to_labels = nn.Linear(h.embedding_dim*2, h.embedding_dim*2)

        # initialize weights
        nn.init.normal_(self.pitch.weight, std=0.02)
        nn.init.normal_(self.speed.weight, std=0.02)
        nn.init.normal_(self.energy.weight, std=0.02)
        nn.init.normal_(self.project_to_labels.weight, std=0.02)

    @staticmethod
    def _upsample(signal, max_frames):
        if signal.dim() == 3:
            bsz, channels, cond_length = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, cond_length = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, cond_length = signal.size()

        signal = signal.unsqueeze(3).repeat(1, 1, 1, max_frames // cond_length)

        # pad zeros as needed (if signal's shape does not divide completely with max_frames)
        reminder = (max_frames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError(
                "Padding condition signal - misalignment between condition features."
            )

        signal = signal.view(bsz, channels, max_frames)
        return signal

    def forward(self, **kwargs):
        x = self.dict(kwargs["code"]).transpose(1, 2)

        dur_losses = 0.0
        if self.dur_predictor:
            if self.multiconds:
                speed = self.speed(kwargs["speed"]).transpose(1, 2)
                speed = self._upsample(speed, x.shape[-1])
                dur_x = torch.cat([x, speed], dim=1)
            else:
                dur_x = x.clone()

            if self.training:
                # assume input code is always full sequence
                uniq_code_feat, uniq_code_mask, dur = process_duration(
                    kwargs["code"], dur_x.transpose(1, 2)
                )
                log_dur_pred = self.dur_predictor(uniq_code_feat)
                log_dur_pred = log_dur_pred[uniq_code_mask]
                log_dur = torch.log(dur + 1)
                dur_losses = F.mse_loss(log_dur_pred, log_dur, reduction="mean")
            elif kwargs.get("dur_prediction", False):
                # assume input code can be unique sequence only in eval mode
                assert dur_x.size(0) == 1, "only support single sample batch in inference"
                log_dur_pred = self.dur_predictor(dur_x.transpose(1, 2))
                dur_out = torch.clamp(
                    torch.round((torch.exp(log_dur_pred) - 1)).long(), min=1
                )
                # B x C x T
                x = torch.repeat_interleave(x, dur_out.view(-1), dim=2)

        if self.multispkr:
            spkr = self.spkr(kwargs["spkr"]).transpose(1, 2)
            spkr = self._upsample(spkr, x.shape[-1])
            # x = torch.cat([x, spkr], dim=1)

        if self.multistyle:
            style = self.style(kwargs["style"]).transpose(1, 2)
            style = self._upsample(style, x.shape[-1])
            # x = torch.cat([x, style], dim=1)
        
        spkr_style = torch.cat([spkr, style], dim=1)

        if self.multiconds:
            pitch = self.pitch(kwargs["pitch"]).transpose(1, 2)
            pitch = self._upsample(pitch, x.shape[-1])
            
            # speed = self.speed(kwargs["speed"]).transpose(1, 2)
            # speed = self._upsample(speed, x.shape[-1])

            energy = self.energy(kwargs["energy"]).transpose(1, 2)
            energy = self._upsample(energy, x.shape[-1])

            # pitch_speed_energy = torch.cat([pitch, speed, energy], dim=1)
            # pitch_speed_energy: [bzs, 384, num_embeddings] --> [bzs, num_embeddings, 384]
            # projected_pitch_speed_energy = self.project_to_labels(pitch_speed_energy.transpose(1,2))
            # projected_pitch_speed_energy = projected_pitch_speed_energy.transpose(1,2)

            pitch_energy = torch.cat([pitch, energy], dim=1)
            projected_pitch_energy = self.project_to_labels(pitch_energy.transpose(1,2))
            projected_pitch_energy = projected_pitch_energy.transpose(1,2)

        labels = torch.add(spkr_style, projected_pitch_energy)
        x = torch.cat([x, labels], dim=1)
        

        return super(DurationCodeGenerator, self).forward(x), dur_losses
