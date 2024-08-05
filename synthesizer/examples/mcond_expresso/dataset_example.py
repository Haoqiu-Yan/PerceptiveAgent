# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan

import torch
import random
import numpy as np
from librosa.util import normalize
from collections import Counter
from dataset import parse_speaker, load_audio, mel_spectrogram, MAX_WAV_VALUE


import os
from pathlib import Path


class MultiConditionsCodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        training_files,
        segment_size,
        code_hop_size,
        n_fft,
        num_mels,
        hop_size,
        win_size,
        sampling_rate,
        fmin,
        fmax,
        fmax_loss=None,
        n_cache_reuse=1,
        device=None,
        input_file=None,
        pad=None,
        multispkr=None,
        multistyle=None,
        speakers=None,
        styles=None,
        multiconds=None,
        pitchs=None,
        speeds=None,
        energys=None,
    ):

        random.seed(1234)

        self.audio_files, self.codes = training_files
        self.segment_size = segment_size
        self.code_hop_size = code_hop_size
        self.sampling_rate = sampling_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.multispkr = multispkr
        self.multistyle = multistyle
        self.pad = pad

        self.multiconds = multiconds

        if self.multispkr:
            if self.multispkr != "from_input_file":
                self.spkr_names = [
                    parse_speaker(f, self.multispkr) for f in self.audio_files
                ]
            else:
                assert (
                    input_file is not None
                ), "input_file is required when multispkr=='from_input_file'"
                with open(input_file) as f:
                    self.spkr_names = [eval(line.strip())["spk"] for line in f]
                assert len(self.spkr_names) == len(self.audio_files)
            
            # Sort the speakers by occurences
            speakers_from_samples = [item[0] for item in Counter(self.spkr_names).most_common()]
            if speakers is None:
                speakers = speakers_from_samples
            
            assert (
                len(speakers_from_samples) <= len(speakers)
            ), "speakers_from_samples is different with speakers.txt provided in advance"       

            self.id_to_spkr = speakers
            self.spkr_to_id = {k: v for v, k in enumerate(self.id_to_spkr)}

        if self.multistyle:
            if self.multistyle != "from_input_file":
                self.style_names = [
                    parse_speaker(f, self.multistyle) for f in self.audio_files
                ]
            else:
                assert (
                    input_file is not None
                ), "input_file is required when multistyle=='from_input_file'"
                with open(input_file) as f:
                    self.style_names = [eval(line.strip())["style"] for line in f]
                assert len(self.style_names) == len(self.audio_files)
           
            # Sort the styles by occurences
            styles_from_samples = [item[0] for item in Counter(self.style_names).most_common()]
            if styles is None:
                styles = styles_from_samples
            
            assert (
                len(styles_from_samples) <= len(styles)
            ), "styles_from_samples is different with styles.txt provided in advance"   

            self.id_to_style = styles
            self.style_to_id = {k: v for v, k in enumerate(self.id_to_style)}

        if self.multiconds:
        
            assert (
                input_file is not None
            ), "input_file is required because multiconds are obtained 'from_input_file'"
            with open(input_file) as f:
                conds_names = [eval(line.strip()) for line in f]
            self.pitch_names = [s["pitch"] for s in conds_names]
            self.speed_names = [s["speed"] for s in conds_names]
            self.energy_names = [s["energy"] for s in conds_names]

            assert len(self.pitch_names) == len(self.audio_files)
            assert len(self.speed_names) == len(self.audio_files)
            assert len(self.energy_names) == len(self.audio_files)
            
            # categories
            if pitchs is None:
                pitchs = [item[0] for item in Counter(self.pitch_names).most_common()]
            if speeds is None:
                speeds = [item[0] for item in Counter(self.speed_names).most_common()]
            if energys is None:
                energys = [item[0] for item in Counter(self.energy_names).most_common()]

            self.id_to_pitch = pitchs
            self.id_to_speed = speeds
            self.id_to_energy = energys

            self.pitch_to_id = {k: v for v, k in enumerate(self.id_to_pitch)}
            self.speed_to_id = {k: v for v, k in enumerate(self.id_to_speed)}
            self.energy_to_id = {k: v for v, k in enumerate(self.id_to_energy)}


    def _sample_interval(self, seqs, seq_len=None):
        N = max([v.shape[-1] for v in seqs])
        if seq_len is None:
            seq_len = self.segment_size if self.segment_size > 0 else N

        hops = [N // v.shape[-1] for v in seqs]
        lcm = np.lcm.reduce(hops)

        # Randomly pickup with the batch_max_steps length of the part
        interval_start = 0
        interval_end = N // lcm - seq_len // lcm

        start_step = random.randint(interval_start, interval_end)

        new_seqs = []
        for i, v in enumerate(seqs):
            start = start_step * (lcm // hops[i])
            end = (start_step + seq_len // lcm) * (lcm // hops[i])
            new_seqs += [v[..., start:end]]

        return new_seqs

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_audio(filename)
            if sampling_rate != self.sampling_rate:
                import resampy

                audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

            if self.pad:
                padding = self.pad - (audio.shape[-1] % self.pad)
                audio = np.pad(audio, (0, padding), "constant", constant_values=0)
            audio = audio / MAX_WAV_VALUE
            audio = normalize(audio) * 0.95
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        # Trim audio ending
        code_length = min(
            audio.shape[0] // self.code_hop_size, self.codes[index].shape[0]
        )
        code = self.codes[index][:code_length]
        audio = audio[: code_length * self.code_hop_size]
        assert (
            audio.shape[0] // self.code_hop_size == code.shape[0]
        ), "Code audio mismatch"

        while audio.shape[0] < self.segment_size:
            audio = np.hstack([audio, audio])
            code = np.hstack([code, code])

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        assert audio.size(1) >= self.segment_size, "Padding not supported!!"
        audio, code = self._sample_interval([audio, code])

        mel_loss = mel_spectrogram(
            audio,
            self.n_fft,
            self.num_mels,
            self.sampling_rate,
            self.hop_size,
            self.win_size,
            self.fmin,
            self.fmax_loss,
            center=False,
        )

        feats = {"code": code.squeeze()}
        if self.multispkr:
            feats["spkr"] = self._get_spkr(index)
        if self.multistyle:
            feats["style"] = self._get_style(index)
        if self.multiconds:
            feats["pitch"] = self._get_pitch(index)
            feats["speed"] = self._get_speed(index)
            feats["energy"] = self._get_energy(index)

        return feats, audio.squeeze(0), str(filename), mel_loss.squeeze()

    def _get_spkr(self, idx):
        spkr_name = self.spkr_names[idx]
        spkr_id = torch.LongTensor([self.spkr_to_id[spkr_name]]).view(1).numpy()
        return spkr_id

    def _get_style(self, idx):
        style_name = self.style_names[idx]
        style_id = torch.LongTensor([self.style_to_id[style_name]]).view(1).numpy()
        return style_id

    def _get_pitch(self, idx):
        pitch_name = self.pitch_names[idx]
        pitch_id = torch.LongTensor([self.pitch_to_id[pitch_name]]).view(1).numpy()
        return pitch_id
    
    def _get_speed(self, idx):
        speed_name = self.speed_names[idx]
        speed_id = torch.LongTensor([self.speed_to_id[speed_name]]).view(1).numpy()
        return speed_id
    
    def _get_energy(self, idx):
        energy_name = self.energy_names[idx]
        energy_id = torch.LongTensor([self.energy_to_id[energy_name]]).view(1).numpy()
        return energy_id

    def __len__(self):
        return len(self.audio_files)


class InferenceMultiConditionsCodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_code_file,
        name_parts=False,
        sampling_rate=None,
        multispkr=None,
        speakers=None,
        forced_speaker=None,
        random_speaker=False,
        random_speaker_subset=None,
        multistyle=None,
        styles=None,
        forced_style=None,
        random_style=False,
        random_style_subset=None,
        pitchs=None,
        speeds=None,
        energys=None,
        multiconds=None,
    ):
        """
        The code file expects each line to be a dictionary
        {"audio": "filename.wav", "hubert": "1 2 2..", "spk": "01", "style": "03"}
        ..
        """

        random.seed(1234)

        if multispkr:
            assert speakers is not None, "speaker list expected for multispkr!"
            assert (
                forced_speaker is None or random_speaker is False
            ), "Cannot force speaker and choose random speaker at the same time"

            self.id_to_spkr = speakers
            self.spkr_to_id = {k: v for v, k in enumerate(self.id_to_spkr)}

            if random_speaker_subset is None or len(random_speaker_subset) == 0:
                random_speaker_subset = speakers

        if multistyle:
            assert styles is not None, "style list expected for multistyle!"
            assert (
                forced_style is None or random_style is False
            ), "Cannot force style and choose random style at the same time"

            self.id_to_style = styles
            self.style_to_id = {k: v for v, k in enumerate(self.id_to_style)}
            if random_style_subset is None or len(random_style_subset) == 0:
                random_style_subset = styles

        if multiconds:
            #TODO: add force multiconds, like force-style
            assert pitchs is not None, "pitch list expected for multipitch!"
            assert speeds is not None, "speed list expected for multispeed!"
            assert energys is not None, "energy list expected for multienergy!"
            
            self.id_to_pitch = pitchs
            self.pitch_to_id = {k: v for v, k in enumerate(self.id_to_pitch)}
            
            self.id_to_speed = speeds
            self.speed_to_id = {k: v for v, k in enumerate(self.id_to_speed)}
        
            self.id_to_energy = energys
            self.energy_to_id = {k: v for v, k in enumerate(self.id_to_energy)}
            

        self.sampling_rate = sampling_rate
        self.multispkr = multispkr
        self.multistyle = multistyle
        self.multiconds = multiconds

        self.audio_files = []
        self.codes = []

        self.spkr_names = []
        self.style_names = []
        self.pitch_names = []
        self.speed_names = []
        self.energy_names = []

        self.output_file_names = []
        with open(input_code_file) as f:
            for line in f:
                content = eval(line)
                # Audio
                audio = content["audio"]
                self.audio_files.append(audio)

                # Code
                code = content["hubert"] if "hubert" in content else content["codes"]
                self.codes.append(code)

                # Speaker
                speaker = None
                if multispkr:
                    if forced_speaker:
                        speaker = forced_speaker
                    elif random_speaker:
                        speaker = random.choice(random_speaker_subset)
                    else:
                        assert "spk" in content, (
                            "Key 'spk' expected in input_code_file when "
                            "not using forced_speaker or random_speaker"
                        )
                        speaker = content["spk"]
                    assert speaker in speakers, (
                        f"Speaker '{speaker}' not in the list of speaker: {speakers}, "
                        "consider forced_speaker or random_speaker"
                    )
                self.spkr_names.append(speaker)

                # Style
                style = None
                if multistyle:
                    if forced_style:
                        style = forced_style
                    elif random_style:
                        style = random.choice(random_style_subset)
                    else:
                        assert "style" in content, (
                            "Key 'style' expected in input_code_file when "
                            "not using forced_style or random_style"
                        )
                        style = content["style"]
                    assert style in styles, (
                        f"Style '{style}' not in the list of style: {styles}, "
                        "consider forced_style or random_style"
                    )
                self.style_names.append(style)

                # Pitch
                pitch = None
                speed = None
                energy = None
                if multiconds:
                    assert "pitch" in content, (
                            "Key 'pitch' expected in input_code_file when "
                            "not using forced_style or random_style"
                        )
                    pitch = content["pitch"]
                    assert pitch in pitchs, (
                        f"Style '{pitch}' not in the list of style: {pitchs}, "
                        "consider forced_style or random_style"
                    )

                    assert "speed" in content, (
                            "Key 'speed' expected in input_code_file when "
                            "not using forced_style or random_style"
                        )
                    speed = content["speed"]
                    assert speed in speeds, (
                        f"Style '{speed}' not in the list of style: {speeds}, "
                        "consider forced_style or random_style"
                    )

                    assert "energy" in content, (
                            "Key 'energy' expected in input_code_file when "
                            "not using forced_style or random_style"
                        )
                    energy = content["energy"]
                    assert energy in energys, (
                        f"Style '{energy}' not in the list of style: {energys}, "
                        "consider forced_style or random_style"
                    )

                self.pitch_names.append(pitch)
                self.speed_names.append(speed)
                self.energy_names.append(energy)

                # Output filename
                if name_parts:
                    parts = Path(audio).parts
                    fname_out_name = os.path.splitext("_".join(parts[-3:]))[0]
                else:
                    fname_out_name = Path(audio).stem
                if multispkr:
                    fname_out_name = fname_out_name + f"_{speaker}"
                if multistyle:
                    fname_out_name = fname_out_name + f"_{style}"
                if multiconds:
                    fname_out_name = fname_out_name + f"_{pitch}_{speed}_{energy}"
                self.output_file_names.append(fname_out_name)

            print(f"Loaded {len(self.audio_files)} files from {input_code_file}!")

        if multispkr:
            if forced_speaker:
                print(f"Force speaker='{forced_speaker}'")
            elif random_speaker:
                print(f"Sample speaker randomly from: {random_speaker_subset}")
            else:
                print(f"Load default speaker from input_code_file")

        if multistyle:
            if forced_style:
                print(f"Force style='{forced_style}'")
            elif random_speaker:
                print(f"Sample style randomly from: {random_style_subset}")
            else:
                print(f"Load default style from input_code_file")

        if multiconds:
            print(f"Load default mconds(pitch,speed,energy) from input_code_file")

    def get_audio(self, path):
        if not os.path.exists(path) or self.sampling_rate is None:
            return None

        audio, sampling_rate = load_audio(path)
        if sampling_rate != self.sampling_rate:
            import resampy

            audio = resampy.resample(audio, sampling_rate, self.sampling_rate)

        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95

        return audio

    def __getitem__(self, index):
        filename = self.audio_files[index]

        audio = self.get_audio(filename)
        code = self.codes[index]
        speaker = self.spkr_names[index]
        style = self.style_names[index]
        pitch = self.pitch_names[index]
        speed = self.speed_names[index]
        energy = self.energy_names[index]
        out_filename = self.output_file_names[index]

        feats = {"code": np.array(list(map(int, code.split())))}
        if self.multispkr:
            feats["spkr"] = np.array([self.spkr_to_id[speaker]])
        if self.multistyle:
            feats["style"] = np.array([self.style_to_id[style]])
        if self.multiconds:
            feats["pitch"] = np.array([self.pitch_to_id[pitch]])
            feats["speed"] = np.array([self.speed_to_id[speed]])
            feats["energy"] = np.array([self.energy_to_id[energy]])

        return feats, audio, str(filename), out_filename

    def __len__(self):
        return len(self.audio_files)
