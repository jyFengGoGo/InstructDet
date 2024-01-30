# Notice: This file is modified by Jiangyan Feng.
# Copyright 2024 Jiangyan Feng.
# 
#    Licensed under the Creative Commons Attribution-NonCommercial International License, Version 4.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://creativecommons.org/licenses/by-nc/4.0/

import argparse
import os
import random
import json
import time

from PIL import ImageDraw, Image
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import gradio as gr
import random
from threading import Thread

import sys
sys.path.append('./')
from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from tqdm import tqdm
import copy
import logging


class MiniGPT4Generator:
    def __init__(self, args):
        self.args = args
        self.cfg = Config(args)
        self.num_beams = 1
        self.temperature = 1
        
        self.petrel_conf = args.petrel_conf
        self.initialized = False

        self.load_prompts()
        self.build_chat()
        
        self.single_only = args.single_only

    def _init_petrel(self):
        if not self.initialized:
            self.client = Client(conf_path=self.petrel_conf)
            self.initialized = True
    

    def load_image(self, image_file):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        elif "s3://" in image_file:
            if not self.initialized:
                self._init_petrel()
            value = self.client.get(image_file, update_cache=True)
            value = memoryview(value)
            filebytes = np.frombuffer(value, dtype=np.uint8)
            buff = io.BytesIO(filebytes)
            with Image.open(buff) as image:
                image = image.convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    

    def load_prompts(self):
        # read task prompts
        prompts = open(self.args.prompt_txt, 'r').readlines()
        self.prompt_list = [_.strip()[23:] for _ in prompts] 

        # texts to replace "red box"
        replace_texts = open(self.args.red_replace_txt, 'r').readlines()
        self.replace_list = [_.strip() for _ in replace_texts]


    def build_chat(self):
        model_config = self.cfg.model_cfg
        # model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        # model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        self.model = model_cls.from_config(model_config).to('cuda')
        # is_train_generate = model_config.train_generate

        vis_processor_cfg = self.cfg.datasets_cfg.ref_coco_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        # chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        self.chat = Chat(self.model, self.vis_processor, device='cuda')


    def visual_prompt(self, image, bbox):
        image_copy = copy.deepcopy(image)
        draw = ImageDraw.Draw(image_copy)
        length = min(bbox[2], bbox[3])
        if length > 100 and length < 200:
            line_width = 5
        elif length > 50 and length < 100:
            line_width = 4
        elif length > 20 and length < 50:
            line_width = 2
        elif length < 20:
            line_width = 1
        else:
            line_width = 6
        draw.rectangle([bbox[0],bbox[1],bbox[0] + bbox[2],bbox[1] + bbox[3]], 
                        outline=(255,0,0), width=line_width)
        return image_copy
    

    @torch.no_grad()
    def _generate(self, image, bbox):
        image_wt_prompt = self.visual_prompt(image, bbox) # draw red box on image
        chat_state_img = CONV_VISION.copy()
        img_list = []
        llm_message = self.chat.upload_img(image_wt_prompt, chat_state_img, img_list)
        prompt_batch = random.sample(self.prompt_list, 3) # generate 3 times
        expressions_vlm = []
        for prompt in prompt_batch:
            chat_state = chat_state_img
            self.chat.ask(prompt, chat_state)
            llm_message = self.chat.answer(conv=chat_state,
                                img_list=img_list,
                                num_beams=self.num_beams,
                                temperature=self.temperature,
                                max_new_tokens=30,
                                max_length=2000)[0]

            # replace "red box" in outputs
            output_instruction = llm_message.split('.')[0]
            replace_text = random.choice(self.replace_list)
            for red_text in ['in the red box', 'In the red box', 'in a red box', 'In a red box', \
                            'A red box', 'a red box', 'the red box', 'The red box', \
                            'Red box', 'red box']:
                output_instruction = output_instruction.replace(red_text, replace_text)
            if output_instruction:
                expressions_vlm.append(output_instruction)
        return expressions_vlm


    def __call__(self, samples):
        for sample in tqdm(samples):
            image_path = sample["filename"]
            image = sample["image"] if "image" in sample else self.load_image(image_path)
            for idx, bbox_info in enumerate(sample["bboxes"]):
                if len(bbox_info["bbox"]) > 1:
                    bbox_info["expressions_vlm"] = []
                    if self.single_only:
                        logging.info("WARN: skipping expression generation for multi-objects(minigpt4)")
                        continue
                    else:
                        bbox_info["expressions_single"] = []
                        for idxx, bbox in enumerate(bbox_info["bbox"]):
                            expressions_vlm = self._generate(image, bbox)
                            bbox_single_info = dict(bbox=[bbox], 
                                                    id=idxx,
                                                    expressions=bbox_info["expressions"],
                                                    expressions_vlm=list(set(expressions_vlm)))
                            bbox_info["expressions_single"].append(bbox_single_info)
                else:
                    bbox = bbox_info["bbox"][0]
                    expressions_vlm = self._generate(image, bbox)
                    bbox_info["expressions_vlm"] = list(set(expressions_vlm))
        return samples
