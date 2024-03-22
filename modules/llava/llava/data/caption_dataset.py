# Notice: This file is created and modified by Jiangyan Feng.
# Copyright 2024 Jiangyan Feng.
# 
#    Licensed under the Creative Commons Attribution-NonCommercial International License, Version 4.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://creativecommons.org/licenses/by-nc/4.0/

import os, json
import numpy as np
import random
from collections import defaultdict
import copy
import torch
from torch.utils.data import Dataset
try:
    import mc
except ImportError:
    pass
from .image_reader import build_image_reader

from llava.conversation import conv_templates, SeparatorStyle
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token
# DEFAULT_IMAGE_TOKEN = "<image>"
# DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
# DEFAULT_IM_START_TOKEN = "<im_start>"
# DEFAULT_IM_END_TOKEN = "<im_end>"


class BaseDataset(Dataset):
    def __init__(self,
                 root_dir,
                 meta_file=None,
                 transform=None,
                 read_from='mc',
                 evaluator=None,
                 petrel_conf=None,
                ):

        super(BaseDataset, self).__init__()

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
        self.read_from = read_from
        self.evaluator = evaluator
        self.petrel_conf = petrel_conf
        self.initialized = False

    def __len__(self):
        """
        Returns dataset length
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Get a single image data: from dataset

        Arguments:
            - idx (:obj:`int`): index of image, 0 <= idx < len(self)
        """
        raise NotImplementedError

    def read_file(self, meta_dict):
        if self.read_from == 'fake':
            if self.initialized:
                filebytes = self.saved_filebytes
            else:
                filebytes = self.saved_filebytes = np.fromfile(meta_dict['filename'], dtype=np.uint8)
                self.initialized = True
        elif self.read_from == 'fs':
            filebytes = np.fromfile(meta_dict['filename'], dtype=np.uint8)
        else:
            raise RuntimeError("unknown value for read_from: {}".format(self.read_from))

        return filebytes
    
    def dump(self, writer, output):
        """
        Dump classification results

        Arguments:
            - writer: output stream
            - output (:obj:`dict`): different for imagenet and custom
        """
        raise NotImplementedError


class CaptionDataset(BaseDataset):
    """
    Dataset that supports llava image caption.

    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_from (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'
        - osg_server (:obj:`str`): '10.198.3.28:30080/components/osg-default/v1'

    Metafile example::
        "{"filename": "n01440764/n01440764_10026.JPEG",
          "label": 0}"
    """
    def __init__(self, root_dir, meta_file, image_processor, tokenizer, transform=None, read_from='fs',
                 evaluator=None, petrel_conf="", image_reader_type='pil', image_half=False, osg_server=None, color_space='RGB', quarys=None,
                 mm_use_im_start_end=True,
                 conv_mode="llava_v1", startidx=None, stride=None,
                 hint=False, easy_hint=False,
                ):
        self.transform = transform
        self.read_from = read_from
        self.evaluator = evaluator
        self.image_reader = build_image_reader(image_reader_type)
        self.image_half = image_half
        self.osg_server = osg_server
        self.initialized = False
        self.color_space = color_space
        self.quarys = quarys
        self.mm_use_im_start_end = mm_use_im_start_end
        self.conv_mode = conv_mode
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.hint = hint
        self.easy_hint = easy_hint
        
        if isinstance(meta_file, list):
            assert len(root_dir) == len(meta_file)
            meta_file = meta_file
            root_dir = root_dir
        else:
            meta_file = [meta_file]
            root_dir = [root_dir]
        self.root_dir = root_dir
        self.meta_file = meta_file
        
        lines = []
        self.num = 0
        self.metas = []
        for i, meta_file_ in enumerate(meta_file):
            with open(meta_file_) as f:
                lines = f.readlines()
            if len(lines) == 1:
                data = json.loads(lines[0])
                filenames = defaultdict(list)
                for img_info in data["images"]:
                    filenames[img_info["file_name"]].append(img_info["expressions"][0])
                self.num += len(filenames)
                for filename in filenames:
                    info = {"filename": os.path.join(self.root_dir[i], filename), 'hints': filenames[filename]}
                    self.metas.append(info)
            else:
                self.num += len(lines)
                for line in lines:
                    info = json.loads(line)
                    info['filename'] = os.path.join(self.root_dir[i], info['filename'])
                    self.metas.append(info)
        if startidx is not None:
            assert stride is not None
            endidx = min(len(self.metas), startidx+stride)
            self.metas = self.metas[startidx:endidx]
            self.num = len(self.metas)
        
        super(CaptionDataset, self).__init__(root_dir=self.root_dir, meta_file=self.meta_file,
                                                read_from=read_from, transform=transform, evaluator=evaluator, petrel_conf=petrel_conf)
    
    def __len__(self):
        return self.num
    
    def get_prompt(self, curr_meta):
        if self.quarys:
            assert isinstance(self.quarys, list)
            qs = random.choice(self.quarys)
        else:
            qs = "Provide a detailed description of the given image"
        
        if self.hint and 'hints' in curr_meta:
            # add specific objects
            qs = qs + f", including objects: {', '.join(curr_meta['hints'])}"
        
        if self.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors=None)
        return input_ids

    def __getitem__(self, idx):
        curr_meta = self.metas[idx]
        filename = curr_meta['filename']
        roi = curr_meta['roi'] if 'roi' in curr_meta else None
        img_bytes = self.read_file(curr_meta)
        img_ori = self.image_reader(img_bytes, filename)
        input_ids = self.get_prompt(curr_meta)
        
        try:
            img = copy.deepcopy(img_ori)
            if roi is not None:
                img = img.crop(roi)
            if self.transform is not None:
                img = self.transform(img)
            img = self.image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
            if self.image_half:
                img = img.half()
            item = {
                'image': img,
                'image_origin': img_ori,
                'input_ids': input_ids,
                'filename': filename,
                'image_id': idx,
                'meta_dict': curr_meta,
            }
            return item
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self))
        
    def dump(self, writer, output):
        filename = output['filename']
        image_id = output['image_id']
        input_ids = output['input_ids']
        outputs = output['outputs']
    
        for _idx in range(len(filename)):
            _output = outputs[_idx]
            _output = _output.strip()
            if _output.endswith(self.stop_str):
                _output = _output[:-len(self.stop_str)]
            _output = _output.strip().replace("\n", "")
            res = {
                'filename': filename[_idx],
                'image_id': int(image_id[_idx]),
                'caption': _output,
            }
            writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()
