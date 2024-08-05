"""
LOCATION: /data1/haoqiuyan/text2unit/code/my_wav2vec_manifest.py
Change from fairseq/examples/wav2vec/wav2vec_manifest.py, (https://github.com/facebookresearch/fairseq) 
- args 'desttsv' (save folder) --> save filename
- args 'ordered' --> write filename in some order
        if not ordered, just skip this command

Updates: 
+ the output tsv can be determined manually
+ write audios in a order, such as 0,1,2,3,4...
-------
Data pre-processing: build vocabularies and binarize training data.
python code/my_wav2vec_manifest.py ./output/vocoder/t2u_100-360-500/small_174mb/valid --desttsv ./output/vocoder/t2u_100-360-500/small_174mb/valid_manifest.tsv --ext wav --valid-percent 0 --ordered
"""

import argparse
import glob
import os
import random

import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.01,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )

    # ---add---
    parser.add_argument(
        "--desttsv", default=".", type=str, metavar="DIR", help="output filename (tsv)"
    )
    parser.add_argument(
        "--ordered", action="store_true", help="write filenames in a order, such as 0,1,2,3,4..."
    )

    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    rand = random.Random(args.seed)

    valid_f = (
        open(os.path.join(args.dest, "valid.tsv"), "w")
        if args.valid_percent > 0
        else None
    )

    with open(args.desttsv, "w") as train_f:
        print(dir_path, file=train_f)

        if valid_f is not None:
            print(dir_path, file=valid_f)
        if args.ordered:
            print(f"Doing ordered is {args.ordered}")
            fpathList = [os.path.realpath(f) for f in glob.glob(search_path, recursive=True)]
            # another method of matching: re.findall("(\d+)",x) extract numeric part in strings
            # fpathListS: [real path, ...]
            fpathListS = sorted(fpathList, key=lambda x: float(os.path.relpath(x, dir_path).split('_')[0]))
            for fname in fpathListS:
                file_path = fname

                if args.path_must_contain and args.path_must_contain not in file_path:
                    continue

                frames = soundfile.info(fname).frames
                dest = train_f if rand.random() > args.valid_percent else valid_f
                print(
                    "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=dest
                )

        else:
            print(f"No Doing ordered is {args.ordered}")
            for fname in glob.iglob(search_path, recursive=True):
                file_path = os.path.realpath(fname)

                if args.path_must_contain and args.path_must_contain not in file_path:
                    continue

                frames = soundfile.info(fname).frames
                dest = train_f if rand.random() > args.valid_percent else valid_f
                print(
                    "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=dest
                )
    if valid_f is not None:
        valid_f.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
