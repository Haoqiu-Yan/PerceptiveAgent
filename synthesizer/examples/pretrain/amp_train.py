# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/jik876/hifi-gan
"""add torch amp"""


import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings(
    action="ignore", message=".*kernel_size exceeds volume extent.*"
)

import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

# when debugging
# import sys
# sys.path.insert(0, "/data/haoqiuyan/describe_speech/code/synthesis/")

from dataset import mel_spectrogram, get_dataset_filelist
from examples.pretrain.dataset_example import MultiSpkrMultiStyleCodeDataset
from examples.pretrain.models_example import MultiSpkrMultiStyleCodeGenerator
from models import (
    MultiPeriodDiscriminator,
    MultiScaleDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)
from utils import (
    plot_spectrogram,
    scan_checkpoint,
    load_checkpoint,
    save_checkpoint,
    build_env,
    AttrDict,
)
from examples.pretrain.utils_example import append_to_config

from torch.cuda.amp import GradScaler, autocast


torch.backends.cudnn.benchmark = True


def train(rank, local_rank, a, h):
    # Initialize distributed training
    if h.num_gpus > 1:
        init_process_group(
            backend=h.dist_config["dist_backend"],
            init_method=h.dist_config["dist_url"],
            rank=rank,
            world_size=h.num_gpus,
        )

    torch.cuda.manual_seed(h.seed)
    device = torch.device("cuda:{:d}".format(local_rank))

    # Read speakers and styles from checkpoint directory
    speakers = None
    if h.get("multispkr", None):
        spkr_file = os.path.join(a.checkpoint_path, "speakers.txt")
        if os.path.exists(spkr_file):
            with open(spkr_file) as f:
                speakers = [line.strip() for line in f]
            if rank == 0:
                print(
                    f"{len(speakers)} speakers loaded from {spkr_file}. First few speakers: {speakers[:5]}"
                )
    styles = None
    if h.get("multistyle", None):
        style_file = os.path.join(a.checkpoint_path, "styles.txt")
        if os.path.exists(style_file):
            with open(style_file) as f:
                styles = [line.strip() for line in f]
            if rank == 0:
                print(
                    f"{len(styles)} styles loaded from {style_file}. First few styles: {styles[:5]}"
                )

    # Create datasets
    training_filelist, validation_filelist = get_dataset_filelist(h)
    trainset = MultiSpkrMultiStyleCodeDataset(
        training_filelist,
        h.segment_size,
        h.code_hop_size,
        h.n_fft,
        h.num_mels,
        h.hop_size,
        h.win_size,
        h.sampling_rate,
        h.fmin,
        h.fmax,
        fmax_loss=h.fmax_for_loss,
        n_cache_reuse=0,
        device=device,
        input_file=h.input_training_file,
        multispkr=h.get("multispkr", None),
        multistyle=h.get("multistyle", None),
        speakers=speakers,
        styles=styles,
    )
    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None
    train_loader = DataLoader(
        trainset,
        num_workers=0,
        shuffle=False,
        sampler=train_sampler,
        batch_size=h.batch_size,
        pin_memory=True,
        drop_last=True,
    )
    if rank == 0:
        validset = MultiSpkrMultiStyleCodeDataset(
            validation_filelist,
            h.segment_size,
            h.code_hop_size,
            h.n_fft,
            h.num_mels,
            h.hop_size,
            h.win_size,
            h.sampling_rate,
            h.fmin,
            h.fmax,
            fmax_loss=h.fmax_for_loss,
            n_cache_reuse=0,
            device=device,
            input_file=h.input_validation_file,
            multispkr=h.get("multispkr", None),
            multistyle=h.get("multistyle", None),
            speakers=trainset.id_to_spkr,
            styles=trainset.id_to_style,
        )
        validation_loader = DataLoader(
            validset,
            num_workers=0,
            shuffle=False,
            sampler=None,
            batch_size=h.batch_size,
            pin_memory=True,
            drop_last=True,
        )

        sw = SummaryWriter(os.path.join(a.checkpoint_path, "logs"))

    # Write speaker & style information to checkpoint dir
    if h.get("multispkr", None) and speakers is None:
        speakers = trainset.id_to_spkr
        if rank == 0:
            print(
                f"{len(speakers)} speakers in the training set. First few speakers: {speakers[:5]}"
            )
            print(f"Writing speaker list to {spkr_file}")
            with open(spkr_file, "w") as f:
                f.write("\n".join(speakers))
    if h.get("multistyle", None) and styles is None:
        styles = trainset.id_to_style
        if rank == 0:
            print(
                f"{len(styles)} styles in the training set. First few styles: {styles[:5]}"
            )
            print(f"Writing style list to {style_file}")
            with open(style_file, "w") as f:
                f.write("\n".join(styles))

    # Pass this information to the generator and write to the config
    if "num_speakers" not in h:
        h["num_speakers"] = len(speakers)
        if rank == 0:
            append_to_config(
                config_file=os.path.join(a.checkpoint_path, "config.json"),
                attributes=[("num_speakers", len(speakers))],
            )
    if "num_styles" not in h:
        h["num_styles"] = len(styles)
        if rank == 0:
            append_to_config(
                config_file=os.path.join(a.checkpoint_path, "config.json"),
                attributes=[("num_styles", len(styles))],
            )

    # Define Generator and Discriminator models
    generator = MultiSpkrMultiStyleCodeGenerator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    # Read last checkpoint from the directory
    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, "g_")
        cp_do = scan_checkpoint(a.checkpoint_path, "do_")

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g["generator"])
        mpd.load_state_dict(state_dict_do["mpd"])
        msd.load_state_dict(state_dict_do["msd"])
        steps = state_dict_do["steps"] + 1
        last_epoch = state_dict_do["epoch"]

    # Distribute models with DDP
    if h.num_gpus > 1:
        generator = DistributedDataParallel(
            generator,
            device_ids=[local_rank],
            find_unused_parameters=("f0_quantizer" in h),
        ).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[local_rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[local_rank]).to(device)

    # Define optimizers
    optim_g = torch.optim.AdamW(
        generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2]
    )
    optim_d = torch.optim.AdamW(
        itertools.chain(msd.parameters(), mpd.parameters()),
        h.learning_rate,
        betas=[h.adam_b1, h.adam_b2],
    )

    # Load optimizer state_dict
    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do["optim_g"])
        optim_d.load_state_dict(state_dict_do["optim_d"])

    # Optimizer schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=h.lr_decay, last_epoch=last_epoch
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=h.lr_decay, last_epoch=last_epoch
    )

    # Start training
    scaler_g = GradScaler()
    scaler_d = GradScaler()

    generator.train()
    mpd.train()
    msd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()

            # Get data
            x, y, _, y_mel = batch
            y = torch.autograd.Variable(y.to(device, non_blocking=False))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=False))
            y = y.unsqueeze(1)
            x = {
                k: torch.autograd.Variable(v.to(device, non_blocking=False))
                for k, v in x.items()
            }

            # Generator forward
            with autocast():
                y_g_hat, dur_losses = generator(**x)

                assert (
                    y_g_hat.shape == y.shape
                ), f"Mismatch in vocoder output shape - {y_g_hat.shape} != {y.shape}"

                # Get mel of generated speech
                y_g_hat_mel = mel_spectrogram(
                    y_g_hat.squeeze(1),
                    h.n_fft,
                    h.num_mels,
                    h.sampling_rate,
                    h.hop_size,
                    h.win_size,
                    h.fmin,
                    h.fmax_for_loss,
                )

            # Discriminator Zero Grad
            optim_d.zero_grad()

            # MPD (MultiPeriodDiscriminator)
            with autocast():
                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
                    y_df_hat_r, y_df_hat_g
                )

                # MSD (MultiScaleDiscriminator)
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
                    y_ds_hat_r, y_ds_hat_g
                )

                # Discriminator Loss
                loss_disc_all = loss_disc_s + loss_disc_f

            # Discriminator Backward Pass
            # loss_disc_all.backward()
            # optim_d.step()
            scaler_d.scale(loss_disc_all).backward()
            scaler_d.step(optim_d)
            scaler_d.update()

            # Generator Zero Grad
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            with autocast():
                loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

                y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
                loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
                loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

                # Generator Loss
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

                # Duration Prediction Loss
                if h.get("dur_prediction_weight", None):
                    loss_gen_all += dur_losses * h.get("dur_prediction_weight", None)

            # Generator Backward Pass
            # loss_gen_all.backward()
            # optim_g.step()
            scaler_g.scale(loss_gen_all).backward()
            scaler_g.step(optim_g)
            scaler_g.update()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print(
                        "Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}".format(
                            steps, loss_gen_all, mel_error, time.time() - start_b
                        )
                    )

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "generator": (
                                generator.module if h.num_gpus > 1 else generator
                            ).state_dict()
                        },
                    )
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(
                        checkpoint_path,
                        {
                            "mpd": (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                            "msd": (msd.module if h.num_gpus > 1 else msd).state_dict(),
                            "optim_g": optim_g.state_dict(),
                            "optim_d": optim_d.state_dict(),
                            "steps": steps,
                            "epoch": epoch,
                        },
                    )

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            x = {
                                k: v.to(device, non_blocking=False)
                                for k, v in x.items()
                            }

                            y_g_hat, dur_losses = generator(**x)
                            y_mel = torch.autograd.Variable(
                                y_mel.to(device, non_blocking=False)
                            )
                            y_g_hat_mel = mel_spectrogram(
                                y_g_hat.squeeze(1),
                                h.n_fft,
                                h.num_mels,
                                h.sampling_rate,
                                h.hop_size,
                                h.win_size,
                                h.fmin,
                                h.fmax_for_loss,
                            )
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio(
                                        "gt/y_{}".format(j),
                                        y[0],
                                        steps,
                                        h.sampling_rate,
                                    )
                                    sw.add_figure(
                                        "gt/y_spec_{}".format(j),
                                        plot_spectrogram(y_mel[0].cpu()),
                                        steps,
                                    )

                                sw.add_audio(
                                    "generated/y_hat_{}".format(j),
                                    y_g_hat[0],
                                    steps,
                                    h.sampling_rate,
                                )
                                y_hat_spec = mel_spectrogram(
                                    y_g_hat[:1].squeeze(1),
                                    h.n_fft,
                                    h.num_mels,
                                    h.sampling_rate,
                                    h.hop_size,
                                    h.win_size,
                                    h.fmin,
                                    h.fmax,
                                )
                                sw.add_figure(
                                    "generated/y_hat_spec_{}".format(j),
                                    plot_spectrogram(
                                        y_hat_spec[:1].squeeze(0).cpu().numpy()
                                    ),
                                    steps,
                                )

                        val_err = val_err_tot / (j + 1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)
                    generator.train()

            steps += 1
            if steps >= a.training_steps:
                break

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print(
                "Time taken for epoch {} is {} sec\n".format(
                    epoch + 1, int(time.time() - start)
                )
            )

    if rank == 0:
        print("Finished training")


def main():
    print("Initializing Training Process..")

    parser = argparse.ArgumentParser()

    parser.add_argument("--group_name", default=None)
    parser.add_argument("--checkpoint_path", default="cp_hifigan")
    parser.add_argument("--config", default="")
    parser.add_argument("--training_epochs", default=2000, type=int)
    parser.add_argument("--training_steps", default=500000, type=int)
    parser.add_argument("--stdout_interval", default=5, type=int)
    parser.add_argument("--checkpoint_interval", default=50000, type=int)
    parser.add_argument("--summary_interval", default=100, type=int)
    parser.add_argument("--validation_interval", default=5000, type=int)
    parser.add_argument("--fine_tuning", default=False, type=bool)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--distributed-world-size", type=int)
    parser.add_argument("--distributed-port", type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, "config.json", a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available() and "WORLD_SIZE" in os.environ:
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = int(os.environ["WORLD_SIZE"])
        h.batch_size = int(h.batch_size / h.num_gpus)
        local_rank = a.local_rank
        rank = a.local_rank
        print("Batch size per GPU :", h.batch_size)
    else:
        rank = 0
        local_rank = 0

    train(rank, local_rank, a, h)


if __name__ == "__main__":
    main()
