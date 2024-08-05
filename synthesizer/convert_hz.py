"""
LOCATION: /data1/haoqiuyan/ws_iwslt/convert_hz.py (1702)
python convert_hz.py --o-path ../datasets/gigas2s/train  --s-path ../datasets/gigas2s/train_16hz --ext flac
"""


from fairseq.data.audio.audio_utils import get_waveform
import soundfile as sf
import os
from pathlib import Path
import argparse
from glob import glob


def convert_hz(origin_path, save_path, sr, ext):    
    files = glob(os.path.join(origin_path, "*.{}".format(ext)))
    print("Loading audios from: ", origin_path)
    for file_path in files: #遍历文件夹
        in_file = file_path.split('/')[-1]
        out_file = file_path.split('/')[-1].split('.')[0] + ".wav"
        # if not os.path.isdir(file):
        save_file = os.path.join(save_path, out_file)
        if os.path.exists(save_file):
            print("exist")
            continue
        else:
            read_file = os.path.join(origin_path, in_file)
            # print("Reading from", origin_path + file)
            wav, sr = get_waveform(
                            read_file,
                            always_2d=False,
                            output_sample_rate=sr,
                            waveform_transforms=None,
                        )

            try:
                sf.write(save_file, wav, sr)
            except sf.LibsndfileError:
                os.mkdir(save_path)
                sf.write(save_file, wav, sr)

            # print("Saving to: ", save_file)
            # break
    print("Saving audios to: ", save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--o-path", required=True, type=Path, help=""
    )
    parser.add_argument(
        "--s-path", required=True, type=Path, help=""
    )
    parser.add_argument(
        "--target-hz", default=16000, type=int, help="target hz"
    )
    parser.add_argument(
        "--ext", default='wav', type=str, help="audio extention"
    )

    args = parser.parse_args()
    convert_hz(args.o_path, args.s_path, args.target_hz, ext=args.ext)


if __name__ == "__main__":
    main()

