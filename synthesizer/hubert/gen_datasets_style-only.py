"""
Construct dataset for vocoder training.
+ input: units, manifest
+ output: {'audio': path, 'hubert': XX, 'duration': xx, 'spk': xx, 'style': xx,}

from: https://github.com/facebookresearch/speech-resynthesis/scripts/tree/main/parse_hubert_codes.py
modified by haoqiu

---
Usage:

python hubert/gen_datasets_style-only.py --codes /speech_synthesis/hubert_kmn_getcodes/units/ljspeech/transcript.units --manifest ./speech_synthesis/hubert_kmn_getcodes/manifest/ljspeech/manifest_all.tsv --outdir /data1/haoqiuyan/ws_iwslt/output/vocoder_in --min-dur 0.01 --tt 0.05 --cv 0.05 --spk-prefix lj
# --spk-from-filename is set as False in default.
"""


import argparse
import random
from pathlib import Path

from tqdm import tqdm


def parse_manifest(manifest):
    """
    manifest contains audio names.
    eg: {"audio": "xxxxxx.wav"}
    """

    audio_files = []

    with open(manifest) as info:
        for line in info.readlines():
            if line[0] == '{':
                sample = eval(line.strip())
                audio_files += [Path(sample["audio"])]
            else:
                audio_files += [Path(line.strip())]

    return audio_files


def split(args, samples):
    """
    read all samples from manifest.
    Args: args.ref_train -- train.manifest
          args.tt -- proportion of split
    """
    
    if int(args.tt) == 0 and int(args.cv) == 0:
        tt = []
        cv = []
        tr = samples
        return tr, cv, tt

    if args.ref_train is not None:
        train_split = parse_manifest(args.ref_train)
        train_split = [x.name for x in train_split]
        val_split = parse_manifest(args.ref_val)
        val_split = [x.name for x in val_split]
        test_split = parse_manifest(args.ref_test)
        test_split = [x.name for x in test_split]
        tt = []
        cv = []
        tr = []

        # parse
        for sample in samples:
            name = Path(sample['audio']).name
            if name in val_split:
                cv += [sample]
            elif name in test_split:
                tt += [sample]
            else:
                tr += [sample]
                assert name in train_split
    else:
        # split
        N = len(samples)
        random.shuffle(samples)
        tt = samples[: int(N * args.tt)]
        cv = samples[int(N * args.tt): int(N * args.tt + N * args.cv)]
        tr = samples[int(N * args.tt + N * args.cv):]

    return tr, cv, tt


def save(outdir, tr, cv, tt):
    # save
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / f'train.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in tr]))
    with open(outdir / f'val.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in cv]))
    with open(outdir / f'test.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in tt]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes', type=Path, required=True, help='File of units, eg: fname|49 1')
    parser.add_argument('--manifest', type=Path, required=True, help='File that contains all audios, with frame numbers')
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--min-dur', type=float, default=None, help='delete files with too short duration')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tt', type=float, default=0.05, help='Proporation of test set')
    parser.add_argument('--cv', type=float, default=0.05, help='Proporation of val set')
    parser.add_argument('--ref-train', type=Path, help='Train manifest that contains all train audios')
    parser.add_argument('--ref-val', type=Path, help='Train manifest that contains all val audios')
    parser.add_argument('--ref-test', type=Path, help='Train manifest that contains all test audios')

    parser.add_argument(
        "--spk-prefix", type=str, default="", help="speaker prefix, usually dataset name such as lj,vctk"
    )
    parser.add_argument(
        "--spk-from-filename", action="store_true", default=False, help="if set, speaker ids are gotten from audio file name "
    )
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.manifest) as f:
        fnames = [l.strip() for l in f.readlines()]
    wav_root = Path(fnames[0])
    fnames = fnames[1:]
# -----------modified--------------------
    with open(args.codes) as f:
        fname_code = {l.split('\t')[0]:l.split('\t')[1] for l in f.readlines()}
        codes = [unit.strip() for unit in fname_code.values()]
        # fnames = [fn.strip()+'.wav' for fn in fname_code.keys()]
# ---------------------------------------
    # parse
    samples = []
    del_file = 0
    
    if args.spk_prefix == 'expresso':
        # for expresso dataset
        for fname_dur, code in tqdm(zip(fnames, codes)):
            sample = {}
            fname, dur = fname_dur.split('\t')

            sample['audio'] = str(wav_root / f'{fname}')
            sample['hubert'] = ' '.join(code.split(' '))
            sample['duration'] = int(dur) / 16000
            # (1) ex04_happy_00194.wav 
            # (2) ex03-ex02_sad-sympathetic_006-ex03_sad_css035.wav_root
            # (3) ex01_default_emphasis_00063.wav
            channel = fname.split('-')[-1]
            spk, style = channel.split('_')[:2]
            sample['style'] = style
            sample['spk'] = spk

            if args.min_dur and sample['duration'] < args.min_dur:
                del_file += 1
                continue

            samples += [sample]
    
    else:
        if args.spk_from_filename:
            for fname_dur, code in tqdm(zip(fnames, codes)):
                sample = {}
                fname, dur = fname_dur.split('\t')

                sample['audio'] = str(wav_root / f'{fname}')
                sample['hubert'] = ' '.join(code.split(' '))
                sample['duration'] = int(dur) / 16000
                sample['style'] = 'default'

                # for vctk dataset
                speaker_id = fname.split('_')[0]
                sample['spk'] = "{}_{}".format(args.spk_prefix, speaker_id)

                if args.min_dur and sample['duration'] < args.min_dur:
                    del_file += 1
                    continue

                samples += [sample]
        
        else:
            for fname_dur, code in tqdm(zip(fnames, codes)):
                sample = {}
                fname, dur = fname_dur.split('\t')

                sample['audio'] = str(wav_root / f'{fname}')
                sample['hubert'] = ' '.join(code.split(' '))
                sample['duration'] = int(dur) / 16000
                sample['style'] = 'default'

                # all audios share one speaker
                sample['spk'] = "{}".format(args.spk_prefix)

                if args.min_dur and sample['duration'] < args.min_dur:
                    del_file += 1
                    continue

                samples += [sample]
    
    
    # split dataset
    tr, cv, tt = split(args, samples)
    save(args.outdir, tr, cv, tt)
    print("DELETE: {} files are deleted, with the SHORTEST limitation {}.".format(del_file, args.min_dur) )


if __name__ == '__main__':
    main()
