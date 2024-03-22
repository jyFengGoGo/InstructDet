# Notice: This file is copied from run_llava.py and is modified by Jiangyan Feng.
# Copyright 2024 Jiangyan Feng.
# 
#    Licensed under the Creative Commons Attribution-NonCommercial International License, Version 4.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://creativecommons.org/licenses/by-nc/4.0/

import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from llava.data.caption_dataset import CaptionDataset
from llava.data.data_loader import build_eval_dataloader, PromptCollate

# from transformers import AutoTokenizer, AutoModelForCausalLM

import os, json
# from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria

import requests
from PIL import Image, ImageDraw
from io import BytesIO
import random

import copy
from tqdm import tqdm
import pandas as pd
import time
import logging

image_describe_quary = [
    "Describe the following image in detail",
    "Provide a detailed description of the given image",
    "Give an elaborate explanation of the image you see",
    "Share a comprehensive rundown of the presented image",
    "Offer a thorough analysis of the image",
    "Explain the various aspects of the image before you",
    "Clarify the contents of the displayed image with great detail",
    "Characterize the image using a well-detailed description",
    "Break down the elements of the image in a detailed manner",
    "Walk through the important details of the image",
    "Portray the image with a rich, descriptive narrative",
    "Narrate the contents of the image with precision",
    "Analyze the image in a comprehensive and detailed manner",
    "Illustrate the image through a descriptive explanation",
    "Examine the image closely and share its details",
    "Write an exhaustive depiction of the given image"
]


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_meta(args, idxls=None):
    meta_file = args.meta_file
    meta_lines = open(meta_file, 'r', encoding='utf-8').readlines()
    if not idxls:
        start = args.startidx
        end = min(len(meta_lines), start+args.stride)
        return meta_lines[start:end]
    else:
        return [meta_lines[idx] for idx in idxls]


def describe_image(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        logging.warning('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode
    
    # load image data
    dataset = CaptionDataset(
        args.image_prefix, 
        args.meta_file,
        image_processor,
        tokenizer,
        read_from=args.read_from,
        image_reader_type='pil',
        petrel_conf=args.petrel_conf if hasattr(args, "petrel_conf") else "~/petreloss.conf",
        image_half=True,
        quarys=image_describe_quary,
        mm_use_im_start_end=model.config.mm_use_im_start_end,
        conv_mode=args.conv_mode,
        startidx=args.startidx,
        stride=args.stride,
        hint=True,
        easy_hint=False,
    )
    
    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]

    collate_fn = PromptCollate(tokenizer, keywords=keywords)
    dataloader = build_eval_dataloader(dataset, collate_fn=collate_fn, batch_size=args.batch_size)
    logging.info(f"LLAVA: Dataset length: {len(dataset)}, Dataloader length: {len(dataloader)}")

    if args.dump and args.outpath:
        fout = open(args.outpath, 'w', encoding='utf-8')
    
    repeat = args.repeat
    lines_with_caption = []
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        image_tensors = batch["image"].cuda()
        input_ids = batch["input_ids"].cuda()
        stopping_criteria = batch["stopping_criteria"]
        meta_dicts = batch["meta_dicts"]
        image_origin = batch["image_origin"] if "image_origin" in batch else None

        outputs_repeat = []
        for k in range(repeat):
            start_time = time.time()
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensors,
                    do_sample=True,
                    temperature=0.5,
                    max_new_tokens=1024,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                    )
            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                logging.warning(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
            batch_time = time.time() - start_time
            
            # save results
            outputs_ = []
            for output in outputs:
                output = output.strip()
                if output.endswith(stop_str):
                    output = output[:-len(stop_str)]
                output = output.strip().replace("\n", "")
                outputs_.append(output)
            outputs_repeat.append(outputs_)
        
        for idx, meta_dict in enumerate(meta_dicts):
            meta_dict["caption"] = "\n".join([outputs_repeat[_][idx] for _ in range(repeat)])
            if image_origin is not None:
                meta_dict["image"] = image_origin[idx]
            lines_with_caption.append(meta_dict)
        
        if args.dump and args.outpath:
            for meta_dict in meta_dicts:
                if "image" in meta_dict:
                    del meta_dict["image"]
                fout.write(json.dumps(meta_dict, ensure_ascii=False)+'\n')
                fout.flush()
    del model
    return lines_with_caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--meta-file", type=str, default='', help="filelist of images")
    parser.add_argument("--image-prefix", type=str, default='')
    parser.add_argument("--outpath", type=str, default='')
    parser.add_argument("--startidx", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1000)
    parser.add_argument("--hint", action='store_true')
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    describe_image(args)
