import random
from typing import Dict, Tuple, List, Union

import torch
import torch.nn as nn
import re
from torch import Tensor
from transformers import LlamaTokenizer
from omegaconf import DictConfig

from imagebind.models.image_bind import imagebind_huge, ImageBindJoiner, ModalityType, replace_joiner_vision
from bubogpt.common.registry import registry
from bubogpt.models.blip2 import BaseModel
from bubogpt.models.modeling_llama import LlamaForCausalLM

# add by haoqiu
import logging
from transformers import AutoProcessor, Wav2Vec2Model
from transformers import GPT2Config
from bubogpt.models.modeling_gpt2 import GPT2LMHeadModel
# from transformers import GPT2LMHeadModel
from imagebind.models.multimodal_formers import disabled_train
from transformers import OpenAIGPTTokenizer, AutoTokenizer

from transformers import BartTokenizer, BartModel, BartForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn import CrossEntropyLoss


def filter_prompt(input_embeds: Dict[str, Tensor], prompt_list: List[str]) -> List[str]:
    if not prompt_list:
        return prompt_list
    input_modal_set = set([k.title() for k in input_embeds if input_embeds[k] is not None])
    prompt_modal_sets = [set(re.findall("<([^<>]+)><ModalityHere></\\1>", prompt)) for prompt in prompt_list]
    results = [prompt_list[i] for i, prompt_modal_set in enumerate(prompt_modal_sets) if
               prompt_modal_set == input_modal_set]
    return results


def arrange_modalities(input_embeds: Dict[str, Tensor], prompt: str) -> List[Tensor]:
    prompt_modalities = re.findall("<([^<>]+)><ModalityHere></\\1>", prompt)
    return [input_embeds[modality.lower()] for modality in prompt_modalities]


def concat_all_embeddings(input_embeds: Dict[str, Tensor], dim: int) -> Tensor:
    embeds = [input_embeds[key] for key in input_embeds if input_embeds[key] is not None]
    return torch.cat(embeds, dim=dim)


def filter_modalities(inputs):
    filtered_inputs = {}

    for k in ModalityType.__dict__.values():
        if k in inputs:
            filtered_inputs[k] = inputs[k]

    return filtered_inputs


@registry.register_model("capsp_qformer_bart")
class CAPSPFromPtQfromerBART(BaseModel):
    """
    My model for CAPtion SPeech.
    ---
    with pretrained qformer, 
    imagebind encoder --> matrix of spectrogram
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "model_type": "model config yaml, path relative in bubogpt directory",
    }

    def __init__(
            self,
            joiner_cfg: DictConfig,
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            freeze_imagebind=True,
            freeze_qformer=False,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=128,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            with_bind_head=False,
            freeze_llm=True,
            use_blip_vision=False,
            proj_model="",
            freeze_wav2vec=True,
            gpt2_config="",
            sampling_rate=16000,
            use_bubogpt_qformer=True,
            bubogpt_q_former_model="/data/haoqiuyan/models/bubogpt/bubogpt_7b.pth",
    ):
        super().__init__()
        assert not low_resource, "Low Resource Mode is Currently Unavailable."

        self.low_resource = low_resource

        import gc
        print('Loading ImageBind')
        self.multimodal_encoder = imagebind_huge(pretrained=True, freeze_imagebind=freeze_imagebind,
                                                 with_head=with_bind_head, use_blip_vision=use_blip_vision)
        print('Loading ImageBind Done')
        gc.collect()

        print('Loading BART')
        # 如果gpt2_config==None, 会报错FileNotFound
        # self.txt_tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        # self.txt_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # self.txt_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False, use_auth_token=True)
        self.txt_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        # because `pad token` dose not exist in LlamaTokenizer
        # but BartTokenizer contains both pad & eos token
        # self.txt_tokenizer.pad_token = self.txt_tokenizer.eos_token
        print("BartTokenizer Finish")
        self.txt_decoder = BartForCausalLM.from_pretrained('facebook/bart-large')
        # self.lm_head = nn.Linear(self.txt_decoder.embed_tokens.embedding_dim, self.txt_tokenizer.vocab_size, bias=False)
        # tips: self.txt_tokenizer.vocab_size == self.txt_decoder.embed_tokens.num_embeddings
        print("BartModel Finish")

        print('Loading BART Done')
        gc.collect()

        print('Loading Q-Former and Adapter/Projector')
        self.multimodal_joiner = ImageBindJoiner(joiner_cfg, output_dim=self.txt_decoder.model.decoder.embed_tokens.embedding_dim)
        if use_bubogpt_qformer:
            load_joiner_qformer(self.multimodal_joiner, bubogpt_q_former_model)
        print('Loading Q-Former and Adapter/Projector Done')
        gc.collect()

        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        self.sampling_rate = sampling_rate

        # print("Preparing Prompts")
        # self.prompt_template = prompt_template
        # if prompt_path:
        #     with open(prompt_path, 'r') as f:
        #         raw_prompts = f.read().splitlines()
        #     self.prompt_list = [prompt_template.format(p) for p in raw_prompts]
        #     print('Load {} training prompts'.format(len(self.prompt_list)))
        #     print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        # else:
        #     self.prompt_list = []
        # print("Preparing Prompts Done")

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            import contextlib
            return contextlib.nullcontext()

    def encode_inputs(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """same with BUBOGPT,
            it takes imagebind encoder"""
        
        with self.maybe_autocast():
            imagebind_outputs = self.multimodal_encoder(inputs)
            llama_inputs = self.multimodal_joiner(imagebind_outputs)
        return llama_inputs
    
    def encode_inputs_wav2vec(self, sp_input: Tensor) -> Tensor:
        """called in CAPSP class,
            it takes wav2vec 2.0 as encoder, 
            where input is 1D and sample rate is required to 16000hz."""
        
        wav2vec_out = {}
        with self.maybe_autocast():
            # imagebind_outputs = self.multimodal_encoder(inputs)
            # llama_inputs = self.multimodal_joiner(imagebind_outputs)

            # ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
            # Is inputs[ModalityType.AUDIO] == ds[0]["audio"]["array"] ?
            sp_values = self.sp_processor(sp_input[ModalityType.AUDIO], return_tensors="pt", sampling_rate=self.sampling_rate).input_values
            sp_values = sp_values.half().cuda()
            sp_values = sp_values.reshape(sp_values.shape[1:])

            wav2vec_out[ModalityType.AUDIO] = self.sp_encoder(sp_values).last_hidden_state
            out = self.multimodal_joiner(wav2vec_out)
        return out

    def prompt_wrap(self, inputs: Dict[str, Tensor], prompt: Union[str, list]) -> Tuple[Tensor, Tensor]:
        if isinstance(prompt, (list, tuple)):
            bs = list(inputs.values())[0].shape[0]
            assert bs == len(prompt)

            return self.batch_prompt_wrap(inputs, prompt)
        elif isinstance(prompt, (str, type(None))):
            return self.single_prompt_wrap(inputs, prompt)
        else:
            raise NotImplementedError(f"Prompt type: {type(prompt)} not supported.")

    def single_prompt_wrap(self, inputs: Dict[str, Tensor], prompt: str) -> Tuple[Tensor, Tensor]:
        if not prompt:
            # 把所有modality的tensor连接成一个一维的
            input_embeds = concat_all_embeddings(inputs, dim=1)
            attns_input = torch.ones(input_embeds.size()[:-1], dtype=torch.long).to(input_embeds.device)
            return input_embeds, attns_input
        input_embeds_list = arrange_modalities(inputs, prompt)
        batch_size = input_embeds_list[0].shape[0]
        prompt_slices = prompt.split('<ModalityHere>')
        prompt_tokens = [self.llama_tokenizer(prompt_slice, return_tensors="pt", add_special_tokens=False)
                         .to(input_embeds_list[0].device) for prompt_slice in prompt_slices]
        prompt_embeds = [self.llama_model.model.embed_tokens(prompt_token.input_ids).expand(batch_size, -1, -1)
                         for prompt_token in prompt_tokens]
        result_embeds = [emb for pair in zip(prompt_embeds[:-1], input_embeds_list)
                         for emb in pair] + [prompt_embeds[-1]]
        wrapped_input_embeds = torch.cat(result_embeds, dim=1)
        wrapped_atts_input = torch.ones(wrapped_input_embeds.size()[:-1],
                                        dtype=torch.long).to(wrapped_input_embeds.device)
        return wrapped_input_embeds, wrapped_atts_input

    def batch_prompt_wrap(self, inputs: Dict[str, Tensor], prompts: List[str]) -> Tuple[Tensor, Tensor]:
        device = list(inputs.values())[0].device
        # This one only works for visual promptinge
        prompt_slices = [prompt.split('<ModalityHere>') for prompt in prompts]
        slice_batch = list(zip(*prompt_slices))

        prompt_tokens = [self.llama_tokenizer(slice,
                                              return_tensors="pt",
                                              add_special_tokens=False,
                                              padding="longest",
                                              truncation=True,
                                              max_length=self.max_txt_len).to(device)
                         for slice in slice_batch]
        prompt_embeds = [self.llama_model.model.embed_tokens(prompt_token.input_ids) for prompt_token in prompt_tokens]
        prompt_masks = [prompt_token.attention_mask for prompt_token in prompt_tokens]

        # NOTE: assuming moalities are the same within a batch
        input_embeds_list = arrange_modalities(inputs, prompts[0])
        input_mask_list = [torch.ones(input_embeds.size()[:-1], dtype=torch.long).to(device) for input_embeds in input_embeds_list]
        result_embeds = [emb for pair in zip(prompt_embeds[:-1], input_embeds_list) for emb in pair] + [prompt_embeds[-1]]
        result_masks = [mask for pair in zip(prompt_masks[:-1], input_mask_list) for mask in pair] + [prompt_masks[-1]]
        wrapped_input_embeds = torch.cat(result_embeds, dim=1)
        wrapped_atts_input = torch.cat(result_masks, dim=1)
        return wrapped_input_embeds, wrapped_atts_input

    def decoder_linear_head(self, decoder_outputs, labels):
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.txt_tokenizer.vocab_size), labels.view(-1))
        
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.last_hidden_state,
        )


    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # filter `inputs` as it may contain informatioins other than modalities, 
        # such as 'text_input', caption of speech
        sp_inputs_dict = filter_modalities(inputs)
        sp_embeds_dict = self.encode_inputs(sp_inputs_dict)
        input_embs = sp_embeds_dict[ModalityType.AUDIO]
        
        self.txt_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in inputs["text_input"]]
        to_regress_tokens = self.txt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=True
        ).to(input_embs.device)

        # add BOS & EOS, by setting add_special_tokens

        # tokens of -100 are not considered when calculating loss
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.txt_tokenizer.pad_token_id, -100
        )

        shift_input_ids = shift_tokens_right(token_ids=to_regress_tokens.input_ids, start_token_id=self.txt_tokenizer.eos_token_id)

        # shift_atten_mask = shift_atten_right(atten_mask=to_regress_tokens.attention_mask)

        with self.maybe_autocast():
            decoder_outputs = self.txt_decoder(
                encoder_hidden_states=input_embs,
                input_ids=shift_input_ids,
                attention_mask=to_regress_tokens.attention_mask,
                return_dict=True,
                labels=targets,
            )       

        loss = decoder_outputs.loss
    
        return {"loss": loss}

    def generate(
            self,
            inputs: Dict[str, Tensor],
            max_length=30,
            min_length=10,
            use_beam=True,
            num_beams=5,
            no_repeat_ngram_size=2, 
            use_topk=False,
            top_k=10,
            use_topk_nucleus=False,
            top_p=0.92,
        ):
        sp_inputs_dict = filter_modalities(inputs)
        sp_embeds_dict = self.encode_inputs(sp_inputs_dict)
        input_embs = sp_embeds_dict[ModalityType.AUDIO]
        
        self.txt_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in inputs["text_input"]]
        to_regress_tokens = self.txt_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=True
        ).to(input_embs.device)

        # add BOS & EOS, by setting add_special_tokens

        # tokens of -100 are not considered when calculating loss
        # targets = to_regress_tokens.input_ids.masked_fill(
        #     to_regress_tokens.input_ids == self.txt_tokenizer.pad_token_id, -100
        # )
    
        # decoding strategy
        if use_beam:    # default
            logging.info("decode with beam")
            decoder_out = self.txt_decoder.generate(
                encoder_hidden_states=input_embs,
                # input_ids=to_regress_tokens.input_ids,
                # attention_mask=to_regress_tokens.attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=max_length,
                min_length=min_length,
                pad_token_id=self.txt_tokenizer.pad_token_id,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size, 
                early_stopping=True,
            )
            
        elif use_topk:
            logging.info("decode with top-k")
            decoder_out = self.txt_decoder.generate(
                encoder_hidden_states=input_embs,
                # input_ids=to_regress_tokens.input_ids,
                # attention_mask=to_regress_tokens.attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=max_length,
                min_length=min_length,
                pad_token_id=self.txt_tokenizer.pad_token_id,
                do_sample=True,
                top_k=top_k
            )
        elif use_topk_nucleus:
            logging.info("decode with top-k (nucleus)")
            decoder_out = self.txt_decoder.generate(
                encoder_hidden_states=input_embs,
                # input_ids=to_regress_tokens.input_ids,
                # attention_mask=to_regress_tokens.attention_mask,
                return_dict_in_generate=True,
                output_scores=True,
                max_length=max_length,
                min_length=min_length,
                pad_token_id=self.txt_tokenizer.pad_token_id,
                do_sample=True,
                top_k=0,
                top_p=top_p
            )
        
        decoder_out_token = decoder_out.sequences
        
        captions = self.txt_tokenizer.batch_decode(decoder_out_token, skip_special_tokens=True)
        # captions = [output[len(self.prompt) :] for output in outputs]

        return [c.replace('.\n','').replace('\n','') for c in captions]


    @classmethod
    def from_config(cls, cfg):
        joiner_cfg = cfg.get("joiner_cfg")
        q_former_model = cfg.get(
            "q_former_model",
            None,
        )
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        # freeze_imagebind = cfg.get("freeze_imagebind", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 128)
        end_sym = cfg.get("end_sym", '\n')
        with_bind_head = cfg.get("with_bind_head", False)
        freeze_llm = cfg.get("freeze_llm", True)
        use_blip_vision = cfg.get("use_blip_vision", False)
        proj_model = cfg.get("proj_model", "")

        freeze_wav2vec = cfg.get("freeze_wav2vec", True)
        gpt2_config = cfg.get("gpt2_config", "")

        bubogpt_q_former_model = cfg.get("bubogpt_q_former_model", "")

        # check sr, since wav2vec2.0 requires 16000hz audios
        # sampling_rate = cfg.get("samping_rate")

        model = cls(
            joiner_cfg=joiner_cfg,
            q_former_model=q_former_model,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            with_bind_head=with_bind_head,
            freeze_llm=freeze_llm,
            use_blip_vision=use_blip_vision,
            proj_model=proj_model,
            freeze_wav2vec=freeze_wav2vec,
            gpt2_config=gpt2_config,
            sampling_rate=16000,
            llama_model=llama_model,
            bubogpt_q_former_model=bubogpt_q_former_model
        )

        
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            if isinstance(ckpt_path, str):
                ckpt_path = [ckpt_path]
            for cur_ckpt_path in ckpt_path:
                print("Load CAPSP model from Checkpoint: {}".format(cur_ckpt_path))
                ckpt = torch.load(cur_ckpt_path, map_location="cpu")
                msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
    

def load_joiner_qformer(joiner:ImageBindJoiner, q_former_model:str):
    """
    load params from bubogpt pretrained model (after 2-stage)
    ---
    pretrained models has output with 4096 embeddings,
    embedding in txt_decoder is changed from 1024 to 4096
    """

    pt_model = torch.load(q_former_model, map_location="cpu")["model"]
    new_state_cls = type(pt_model)
    new_state = new_state_cls({k.split('.',1)[1]:v for k,v in pt_model.items()})
    # delete vision-related params
    new_state.pop("modality_post_projectors.vision.fc.weight")
    new_state.pop("modality_post_projectors.vision.fc.bias")
    # 不加载linear params, 重新学
    new_state.pop("modality_post_projectors.audio.fc.weight")
    new_state.pop("modality_post_projectors.audio.fc.bias")

    msg = joiner.load_state_dict(new_state, strict=False)
    # joiner.modality_qformers[ModalityType.AUDIO].load_Qformer(q_former_model)


def shift_tokens_right(token_ids, start_token_id):
    """
    ref : /data/haoqiuyan/anaconda3/envs/hug38/lib/python3.8/site-packages/transformers/models/bart/modeling_bart.py
    fun shift_tokens_right()
    """
    shifted_input_ids = token_ids.new_zeros(token_ids.shape)
    shifted_input_ids[:, 1:] = token_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = start_token_id

    return shifted_input_ids


def shift_atten_right(atten_mask):

    return shift_tokens_right(atten_mask, 1)