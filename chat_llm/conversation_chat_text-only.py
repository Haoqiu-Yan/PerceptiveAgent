"""
TEXT only as input, without caption.
---
Usage

python code/openai/conversation_chat_text-only.py --id-convs ./framework/meld/id_convs.txt --audio-asr-caption ./framework/meld/audio_asr_caption.txt --save-root ./framework/openai/test 

"""


import os
import re
import json
import argparse
from pathlib import Path

from openai import OpenAI
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored


GPT_MODEL = "gpt-3.5-turbo-1106"


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_with_response(messages):

    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    response = client.chat.completions.create(
        messages=messages,
        model=GPT_MODEL,
        )

    return response


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "tool": "magenta",
    }
    
    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant: {message['function_call']}\n", role_to_color[message["role"]]))
        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant: {message['content']}\n", role_to_color[message["role"]]))
        elif message["role"] == "tool":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))


def get_samples(conv_seq_file, asr_caption_pairs_file):
    with open(conv_seq_file, 'r') as cf:
        history_currentL = [(line.split('\t')[0],line.split('\t')[1].strip()) for line in cf.readlines()]
    
    with open(asr_caption_pairs_file, 'r') as af:
        audio_asr_captionD ={line.split('\t')[0]: (line.split('\t')[1], line.split('\t')[2].strip()) for line in af.readlines()}
    
    
    for historys, current in history_currentL:
        samples = {'history': [], 'current': ''}
        historysL = historys.split(',')
        last_speaker = ""
        for history in historysL:
            asr_caption, speaker = get_asr_caption(history, audio_asr_captionD)
            if speaker == last_speaker:
                # merge into one
                samples['history'][-1] = ("{} {}".format(samples['history'][-1][0], asr_caption[0]),\
                                          samples['history'][-1][1])
            else:
                samples['history'].append(asr_caption)
            last_speaker = speaker
            
        samples['current'] =  get_asr_caption(current, audio_asr_captionD)
        
        yield samples


def get_asr_caption(audio_name, audio_asr_captionD):
    """
    audio_name: dev-dia0_utt0-Phoebe-sadness.wav
    audio_asr_captionD: {dev-dia0_utt0.wav: (asr, caption)}
    """
    audio_prefix = re.match(r'(.*-dia.*_utt.*?)-.*wav', audio_name).group(1)
    spk = audio_name.split('-')[-2]
    return audio_asr_captionD[audio_prefix + '.wav'], spk


def save_all_json(response, save_path):
    with open(save_path, "a+") as sf:
        sf.write(json.dumps(response) + '\n')
    print(f"Writing ALL to {save_path}")


def process_gpt_response(response):
    return response


def save_gen_gt(gt_outputL, gen_outputL, save_root):
    with open(os.path.join(save_root, 'gt_asr_caption.txt'), "a+") as gtf:
        for s in gt_outputL:
            print(s.replace('\n', ''), file=gtf)

    with open(os.path.join(save_root, 'gen_asr_caption.txt'), "a+") as genf:
        for s in gen_outputL:
            print(s.replace('\n', ''), file=genf)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--id-convs", required=True, type=Path, help="id_convs.txt, each line contains history + current name"
    )
    parser.add_argument(
        "--audio-asr-caption", required=True, type=Path, help="audio_asr_caption.txt, all audios are saved as name + asr + caption"
    )
    parser.add_argument(
        "--save-root", required=True, type=Path, help="folder to save gt && gen asr + caption"
    )
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    
    args = get_parser()
    if not os.path.exists(args.save_root):
        os.mkdir(args.save_root)
    sample_iterator = get_samples(args.id_convs, args.audio_asr_caption)
    
    gt_outputL = []
    gen_outputL = []

    save_interval = 100

    # sample_test = [416, 653, 2761, 3091, 4623]
    sample_count = 1
    for sample in sample_iterator:
        if sample_count < 601:
            print(f"SKIP {sample_count}")
            sample_count += 1
            continue
        # if sample_count not in sample_test:
        #     sample_count += 1
        #     continue
        print(f"\n\nProcessing {sample_count}")

        num = 1
        input_content = ""
        for asr,caption in sample['history']:
            if num % 2 == 0:
                input_content += "Speaker B: {}\n\n".format(asr)
            else:
                input_content += "Speaker A: {}\n\n".format(asr)
            num += 1
        
        if num % 2 == 0:
            input_content += "Speaker B: "
        else:
            input_content += "Speaker A: "
        
        print(input_content)
        
        messages=[
                    {
                        "role": "system",
                        "content": f"You are the last speaker in the following daily dialogue.\nYou MUST give a response depending on the dialogue history. You MUST keep response as short as possible.",
                    },
                    {
                        "role": "user",
                        "content": f"Speaker A: My specimen is deposited into the container in the room. Janice! You're not... gone?\n\nSpeaker B: Oh! Sid is still in his room. So did you do it? Did you make your deposit?\n\nSpeaker A: Yeah! yeah... The hard part is over!\n\nSpeaker B: ",

                    },
                    {
                        "role": "assistant",
                        "content": f"That's not the hard part honey! The hard part is what comes next, I mean aren't you worried about the results?",

                    },
                    {
                        "role": "user",
                        "content": f"{input_content}",

                    },
                    
                ]
        
        try:
            chat_response = chat_with_response(messages)
            chat_response = chat_response.json()
            print(chat_response)
        except Exception:
            if len(gen_outputL) > 0:
                save_gen_gt(gt_outputL, gen_outputL, args.save_root)
                print(f"[PIPE] Partly saving gt_outputL & gen_outputL ({len(gen_outputL)})")
            raise

        chat_response = json.loads(chat_response)

        try:
            chat_content = chat_response["choices"][0]["message"]["content"]
        except KeyError:
            if len(gen_outputL) > 0:
                save_gen_gt(gt_outputL, gen_outputL, args.save_root)
                print(f"[PIPE] Partly saving gt_outputL & gen_outputL ({len(gen_outputL)})")
            raise KeyError(f"[ERROR] Processing sample {sample_count}: Empty Response!")
        
        cur_asr, cur_caption = sample['current']
        gt_outputL.append("{}\t{}".format(sample_count, cur_asr))

        try:
            gen_asr = process_gpt_response(chat_content)
            gen_outputL.append("{}\t{}".format(sample_count, gen_asr))
        except AttributeError:
            gen_outputL.append("[EXCEPTION]{}".format(chat_content))
        
        
        if (sample_count % save_interval == 0):
            save_gen_gt(gt_outputL, gen_outputL, args.save_root)
            print(f"[PIPE] sample_count: {sample_count}")
            print(f"[PIPE] save gt_outputL (len {len(gt_outputL)})")
            print(f"[PIPE] save gen_outputL (len {len(gen_outputL)})")
            # clear the pipeline
            gt_outputL, gen_outputL = [], []
        
        
        sample_count += 1
        # if sample_count >= 101:
        #     break

        # time.sleep(1)
    
    if len(gen_outputL) > 0:
        save_gen_gt(gt_outputL, gen_outputL, args.save_root)
        print(f"[PIPE] Partly saving gt_outputL & gen_outputL ({len(gen_outputL)})")

