# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Modified from https://github.com/facebookresearch/textlesslib/tree/main/tools/distributed_transcribe/transcribe.py
"""


import torch.distributed as distr
import torch
import pathlib
from tools.distributed_transcribe.data_handler import ManifestDataset
from tools.distributed_transcribe.distributed import init_distributed_context
import numpy as np
import os
import logging
import tqdm
from pathlib import Path
import torchaudio

from textless.data.speech_encoder import SpeechEncoder
from textless.data.encodec_reader import EncodecFeatureReader
from textless.data.seamless_feature_reader import Wav2vecFeatureReader

logger = logging.getLogger(__name__)

SAMPLING_RATE = 16_000


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab_size",
        default=100,
        type=int,
        help="Quantization codebook vocabulary size",
    )
    parser.add_argument(
        "--dense_model", default="hubert-base-ls960", help="Dense model to be used"
    )
    parser.add_argument(
        "--quantizer_model", default="kmeans", help="Quantizer model to be used"
    )

    parser.add_argument(
        "--manifest", required=True, help="Path to the dataset manifest file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output files. Pseudo-units and duration (if requested) streams will be stored in files with .units and .durations suffixes, respectively",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="if set, consecutive repeats of the same pseudo-unit are collapsed ('1 2 2 2 3' becomes '1 2 3')",
    )
    parser.add_argument(
        "--durations",
        action="store_true",
        help="if set, the token durations stream is produced",
    )
    parser.add_argument(
        "--f0s",
        action="store_true",
        help="if set, the F0 stream is produced",
    )
    parser.add_argument(
        "--preserve_name",
        action="store_true",
        help="If set, the transcript contains names of the audio files",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=" ",
        help="Separator between pseudo-unit tokens",
    )
    parser.add_argument(
        "--codec",
        action="store_true",
    )
    parser.add_argument(
        "--seamless",
        action="store_true",
    )

    parser.add_argument("--distributed_port", type=int, default=58554)

    args = parser.parse_args()
    logger.info(f"Launched with args: {args}")

    return args


def worker_shard_path(fname, suffix, worker_id) -> pathlib.Path:
    return pathlib.Path(fname).with_suffix(f".{suffix}_partial_{worker_id}")


def transcribe(args, rank, world_size):
    dataset = ManifestDataset(args.manifest)

    if args.codec:
        bandwidth = 12
        # download first
        speech_encoder = EncodecFeatureReader(
            bandwidth=bandwidth, 
            repository=Path(os.getenv('TORCH_HOME')) / "hub" / "encodec"
        )
    elif args.seamless:
        speech_encoder = Wav2vecFeatureReader(
            "xlsr2_1b_v2", 
            "https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
            layer=35
        )
    else:
        # single process download
        if world_size > 1:
            if rank == 0:
                SpeechEncoder.by_name(
                    dense_model_name=args.dense_model,
                    quantizer_model_name=args.quantizer_model,
                    vocab_size=args.vocab_size,
                    deduplicate=args.deduplicate,
                    need_f0=args.f0s,
                )
            distr.barrier()
        speech_encoder = SpeechEncoder.by_name(
            dense_model_name=args.dense_model,
            quantizer_model_name=args.quantizer_model,
            vocab_size=args.vocab_size,
            deduplicate=args.deduplicate,
            need_f0=args.f0s,
        ).cuda()

    output_files = {
        "units": open(worker_shard_path(args.output, "units", rank), "w")
        if not args.codec else None,
        "codec": open(worker_shard_path(args.output, "codec", rank), "w")
        if args.codec else None,
        "durations": None
        if not args.durations
        else open(worker_shard_path(args.output, "durations", rank), "w"),
        "f0s": None
        if not args.f0s
        else open(worker_shard_path(args.output, "f0s", rank), "w"),
    }

    # DistributedSampler will pad the dataloader to be divisible
    # by the number of workers, which we do not want so we iterate directly
    for i in tqdm.tqdm(range(rank, len(dataset), world_size)):
        audio_path, name, path = dataset[i]
                
        if args.codec:
            if ((dataset.root).parent / f'meta24khz_{bandwidth}kpbs_codec' / dataset.root.stem / path.with_suffix('.npy')).exists():
                continue
        if args.seamless:
            if ((dataset.root).parent / 'xlsr2_unit' / dataset.root.stem / path.with_suffix('.npy')).exists():
                continue
        
        waveform, sr = torchaudio.load(str(audio_path))
        assert sr == 16_000
        waveform = waveform.squeeze(0)
        
        encoded = speech_encoder(waveform)
        
        if args.codec:
            stream_name = "codec"
            stream = encoded['codec'].shape[-1]
            os.makedirs(str((dataset.root).parent / f'meta24khz_{bandwidth}kpbs_codec' / dataset.root.stem / path.parent), exist_ok=True)
            np.save(str((dataset.root).parent / f'meta24khz_{bandwidth}kpbs_codec' / dataset.root.stem / path.with_suffix('.npy')), encoded["codec"].numpy())
            stream = f"{path.with_suffix('')}.npy\t{stream}" if args.preserve_name else stream
            print(stream, file=output_files[stream_name])
            continue
        
        if args.seamless:
            stream_name = "units"
            stream = encoded['units'].shape[-1]
            os.makedirs(str((dataset.root).parent / "xlsr2_unit" / dataset.root.stem / path.parent), exist_ok=True)
            np.save(str((dataset.root).parent / 'xlsr2_unit' / dataset.root.stem / path.with_suffix('.npy')), encoded["units"].numpy())
            stream = f"{path.with_suffix('')}.npy\t{stream}" if args.preserve_name else stream
            print(stream, file=output_files[stream_name])
            continue

        stream_names = ["units"]
        if args.f0s:
            stream_names += ["f0s"]
        if args.durations:
            stream_names += ["durations"]

        for stream_name in stream_names:
            stream = encoded[stream_name]
            stream = [str(int(x)) for x in stream.tolist()]
            stream = args.separator.join(stream)

            stream = f"{path.with_suffix('')}\t{stream}" if args.preserve_name else stream
            print(stream, file=output_files[stream_name])

    for fout in output_files.values():
        if fout:
            fout.close()


def main(args):
    context = init_distributed_context(args.distributed_port)
    logger.info(f"Distributed context {context}")

    n_gpus = torch.cuda.device_count()
    with torch.cuda.device(context.local_rank % n_gpus):
        transcribe(args, context.rank, context.world_size)

    if context.world_size > 1:
        distr.barrier()

    if context.is_leader:
        if args.codec:
            merge_files(args.output, "codec", context.world_size)
        elif args.seamless:
            merge_files(args.output, "units", context.world_size)
        else:
            generated_streams = ["units"]
            if args.durations:
                generated_streams += ["durations"]
            if args.f0s:
                generated_streams += ["f0s"]

            for stream_name in generated_streams:
                merge_files(args.output, stream_name, context.world_size)


def merge_files(full_output, suffix, n_workers):
    output = full_output + f".{suffix}"
    with open(output, "w") as full:
        for worker_id in range(n_workers):
            partial_path = worker_shard_path(full_output, suffix, worker_id)
            partial = open(partial_path, "r")
            for line in partial:
                print(line.strip(), file=full)
            partial_path.unlink()


if __name__ == "__main__":
    args = get_args()
    main(args)
