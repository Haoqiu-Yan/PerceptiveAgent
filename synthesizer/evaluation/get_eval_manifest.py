"""
get eval_f0 input:

"ref"   "syn"
/path/to/ex04_default_00340_gt.wav /path/to/ex04_default_00340_gen.wav
/path/to/ex04_default_00341_gt.wav /path/to/ex04_default_00341_gen.wav
---
Usage:

python code/synthesis/evaluation/get_eval_manifest.py  --gen-extension gen.wav  --from-dir speech_synthesis/evaluation/dev/audios  --manifest speech_synthesis/evaluation/dev/gt_gen_manifest.csv
"""


import os
import csv
from pathlib import Path


def find_all_files(path_dir, extension):
    out = []
    for root, dirs, filenames in os.walk(path_dir):
        for f in filenames:
            if f.endswith(extension):
                out.append(os.path.join(root, f))
    return out


def main(args):
    """
    `uid syn ref text`
    """
    print(
            f"Finding all gen audio files with extension '{args.gen_extension}' from {args.from_dir}..."
        )
    files = find_all_files(args.from_dir, args.gen_extension)
    files = [os.path.abspath(path) for path in files]
    print(f"Done! Found {len(files)} gen files.")

    headers = ["ref", "syn"]
    with open(args.manifest, 'w') as mf:
        print("\t".join(headers), file=mf)
        for gen in files:
            gt = gen.replace("gen.wav", "gt.wav")
            print("{}\t{}".format(gt, gen), file=mf)

    print(f"Saving to {args.manifest}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gen-extension",  help="gen.wav"
    )
    parser.add_argument(
        "--from-dir",
        help="folder of audios"
    )
    parser.add_argument(
        "--manifest", help="save manifest as"
    )
    
    args = parser.parse_args()

    main(args)
