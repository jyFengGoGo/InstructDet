import os
import torch
import json
import clip
from PIL import Image, ImageDraw
import copy
import time
from tqdm import tqdm
import numpy as np
import io
import argparse
import random
from petrel_client.client import Client


class CLIPFilter:
    def __init__(self, config):
        self.config = config
        self.alpha = config.alpha
        self.tlr = config.tlr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.load_model()
        self.initialized = False

    def _init_petrel(self):
        if not self.initialized:
            self.client = Client(conf_path=self.config.petrel_conf)
            self.initialized = True

    def load_model(self):
        self.model, self.preprocess = clip.load(self.config.model_path, device=self.device)

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
    
    def compute_score(self, image, bbox, caption):  
        # input: all captions for the bbox
        origin_image = copy.deepcopy(image)
        draw_image = copy.deepcopy(image)
        draw = ImageDraw.Draw(draw_image)
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
        draw.ellipse([bbox[0],bbox[1],bbox[0] + bbox[2],bbox[1] + bbox[3]], outline=(255,0,0), width=line_width)

        draw_image = self.preprocess(draw_image).unsqueeze(0).to(self.device)   # image feature
        origin_image = self.preprocess(origin_image).unsqueeze(0).to(self.device)
        text = clip.tokenize(caption).to(self.device)
        with torch.no_grad():
            logits_per_image, logits_per_text = self.model(draw_image, text)
            logits_per_image_o, logits_per_text_o = self.model(origin_image, text)
            probs = logits_per_image.cpu().numpy()
            probs_o = logits_per_image_o.cpu().numpy()
        return probs, probs_o
    
    def holistic_score_1(self, score_local, score_global):
        return self.alpha * score_local - score_global   
    
    def score_filter_separa(self, holistic_score_e, holistic_score_o):
        good_caption_index = []
        baseline = min(holistic_score_o) - self.tlr
        return np.argwhere(holistic_score_e > baseline)[:, 0]
    
    def _filter(self, image, bboxes, caption_exp, caption_ori):
        caption_exp_drop = []
        for bbox in bboxes:
            if len(caption_exp) > 0:
                score_local_e, score_global_e = self.compute_score(image, bbox, caption_exp)
                score_local_o, score_global_o = self.compute_score(image, bbox, caption_ori)
                
                holistic_score_e = self.holistic_score_1(score_local_e[0], score_global_e[0])
                holistic_score_o = self.holistic_score_1(score_local_o[0], score_global_o[0])

                good_caption_index = self.score_filter_separa(holistic_score_e, holistic_score_o)
                if len(good_caption_index) != len(caption_exp):
                    a = [i for i in range(len(caption_exp))]
                    bad_caption_index = list(set(a)-set(good_caption_index))
                    for id in bad_caption_index:
                        caption_exp_drop.append(caption_exp[id])
                    caption_exp = [caption_exp[idx] for idx in good_caption_index]
            else:
                break
        return caption_exp, caption_exp_drop
    
    @torch.no_grad()
    def cluster_filter(self, meta_dicts, exptypes=[]):
        meta_dicts_filtered = []
        for meta_dict in meta_dicts:
            image = meta_dict["image"] if "image" in meta_dict else self.load_image(meta_dict["filename"])
            meta_dict_filtered = copy.deepcopy(meta_dict)
            for index, cluster in enumerate(meta_dict["clusters"]):
                bbox_ids = cluster["bbox_ids"]
                for exptype in exptypes:
                    caption_exp = cluster[exptype] if exptype in cluster else []
                    exp_drops = []
                    for bbox_id in bbox_ids:
                        box_info = meta_dict["bboxes"][bbox_id]
                        caption_ori = box_info["expressions"]
                        exp_keep, exp_drop = self._filter(image, box_info["bbox"], caption_exp, caption_ori)
                        caption_exp = exp_keep
                        exp_drops += exp_drop
                    meta_dict_filtered["clusters"][index][exptype] = exp_keep
                    meta_dict_filtered["clusters"][index][f"{exptype}_filtered"] = exp_drops
            meta_dicts_filtered.append(meta_dict_filtered)
        return meta_dicts_filtered

    @torch.no_grad()
    def __call__(self, meta_dicts, exptypes=[]):
        meta_dicts_filtered = []
        for meta_dict in meta_dicts:
            image = meta_dict["image"] if "image" in meta_dict else self.load_image(meta_dict["filename"])
            meta_dict_filtered = copy.deepcopy(meta_dict)
            for index, box_info in enumerate(meta_dict["bboxes"]):
                caption_ori = box_info["expressions"]
                for exptype in exptypes:
                    if exptype != "expressions_single":
                        exptypef = f"{exptype}_filtered"
                        caption_exp = box_info[exptype] if exptype in box_info else []
                        for exp in caption_ori:
                            if exp in caption_exp:
                                caption_exp.remove(exp)
                        exp_keep, exp_drop = self._filter(image, box_info["bbox"], caption_exp, caption_ori)
                        meta_dict_filtered["bboxes"][index][exptype] = exp_keep
                        meta_dict_filtered["bboxes"][index][exptypef] = exp_drop
                    else:
                        expsingle = box_info[exptype] if exptype in box_info else [] # list of dicts
                        meta_dict_ = copy.deepcopy(meta_dict)
                        meta_dict_["bboxes"] = expsingle
                        meta_dict_filtered_ = self([meta_dict_], exptypes=["expressions_vlm"])
                        meta_dict_filtered["bboxes"][index][exptype] = meta_dict_filtered_[0]["bboxes"]
            meta_dicts_filtered.append(meta_dict_filtered)
        return meta_dicts_filtered


def remove_conflict(expressions):
    expressions_copy = copy.deepcopy(expressions)
    for exp in expressions:
        if "left" in exp:
            if exp.replace("left", "right") in expressions_copy:
                expressions_copy.remove(exp)
                expressions_copy.remove(exp.replace("left", "right"))
                print(f"WARN: conflict expressions found. remove {exp}...")
        elif "right" in exp:
            if exp.replace("right", "left") in expressions_copy:
                expressions_copy.remove(exp)
                expressions_copy.remove(exp.replace("right", "left"))
                print(f"WARN: conflict expressions found. remove {exp}...")
    return expressions_copy
