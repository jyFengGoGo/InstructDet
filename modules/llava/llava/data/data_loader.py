# Notice: This file is created and modified by Jiangyan Feng.
# Copyright 2024 Jiangyan Feng.
# 
#    Licensed under the Creative Commons Attribution-NonCommercial International License, Version 4.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://creativecommons.org/licenses/by-nc/4.0/

from torch.utils.data import DataLoader
import torch

from llava.mm_utils import KeywordsStoppingCriteria

class PromptCollate:
    """Puts each data field into a tensor with outer dimension batch size"""
    def __init__(self, tokenizer, keywords=[]):
        self.tokenizer = tokenizer
        self.keywords = keywords
    
    def _batch_pad(self, samples, pad_token_id=0, padding_side='left'):
        max_width = max([len(_) for _ in samples])
        if padding_side == "right":
            samples = [_ + [pad_token_id]*(max_width - len(_)) for _ in samples]
        elif padding_side == "left":
            samples = [[pad_token_id]*(max_width - len(_)) + _  for _ in samples]
        return torch.as_tensor(samples)

    def __call__(self, batch):
        '''
        item = {
                'image': img,
                'image_origin': img_ori,
                'input_ids': input_ids,
                'filename': filename,
                'image_id': idx,
                'meta_dict': curr_meta,
            }
        '''
        new_batch = {}
        new_batch["filename"] = [entry["filename"] for entry in batch]
        new_batch["image"] = torch.stack([entry["image"] for entry in batch], 0).half()
        new_batch["image_origin"] = [entry["image_origin"] for entry in batch]
        new_batch["input_ids"] = self._batch_pad([entry["input_ids"] for entry in batch])
        new_batch["image_id"] = [entry["image_id"] for entry in batch]
        new_batch["stopping_criteria"] = KeywordsStoppingCriteria(self.keywords, self.tokenizer, new_batch["input_ids"])
        new_batch["meta_dicts"] = [entry["meta_dict"] for entry in batch]
        return new_batch
    

def build_eval_dataloader(dataset, batch_size=8, num_workers=4, pin_memory=True, collate_fn=None):
    # build dataloader
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        collate_fn=collate_fn
                        )
    return loader
