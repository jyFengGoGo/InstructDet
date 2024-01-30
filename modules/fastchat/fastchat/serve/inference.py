# Notice: This file is modified by Jiangyan Feng.
# Copyright 2024 Jiangyan Feng.
# 
#    Licensed under the Creative Commons Attribution-NonCommercial International License, Version 4.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        https://creativecommons.org/licenses/by-nc/4.0/



"""Inference for FastChat models."""
import abc
import gc
import math
from typing import Iterable, Optional
import sys, os, json
import time
import warnings

import psutil
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from fastchat.conversation import get_conv_template, SeparatorStyle
from fastchat.model.model_adapter import load_model, get_conversation_template
from fastchat.model.chatglm_model import chatglm_generate_stream
from fastchat.model.falcon_model import falcon_generate_stream
from fastchat.modules.gptq import GptqConfig
from tqdm import tqdm
import random
import copy
import re

import logging


def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def partial_stop(output, stop_str):
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


@torch.inference_mode()
def generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]

    if model.config.is_encoder_decoder:
        encoder_output = model.encoder(
            input_ids=torch.as_tensor([input_ids], device=device)
        )[0]
        start_ids = torch.as_tensor(
            [[model.generation_config.decoder_start_token_id]],
            dtype=torch.int64,
            device=device,
        )

    past_key_values = out = None
    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=start_ids,
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                )
                logits = model.lm_head(out[0])
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
            past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model.decoder(
                    input_ids=torch.as_tensor([[token]], device=device),
                    encoder_hidden_states=encoder_output,
                    use_cache=True,
                    past_key_values=past_key_values,
                )

                logits = model.lm_head(out[0])
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")

            # prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate_complete(model, tokenizer, params, device, context_len=2048
):
    prompt = params["prompt"]
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer([prompt]).input_ids
    input_echo_len = len(input_ids)
    
    output_ids = model.generate( 
            torch.as_tensor(input_ids, device=device), 
            do_sample=True, 
            temperature=temperature, 
            max_new_tokens=max_new_tokens
        )
    if model.config.is_encoder_decoder:
        output_ids = output_ids[0]
    else:
        output_ids = output_ids[0][len(input_ids[0]) :]
    outputs = tokenizer.decode(
        output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""


def normbbox(box, w, h):
    if not isinstance(box[0], list):
        x, y, wi, hi = box
        return [round(x/w,2), round(y/h,2), round((x+wi)/w,2), round((y+hi)/h,2)]
    else:
        bboxes = []
        for b in box:
            x, y, wi, hi = b
            bboxes.append([round(x/w,2), round(y/h,2), round((x+wi)/w,2), round((y+hi)/h,2)])
        return bboxes


def gen_instruction(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    chatio: ChatIO,
    gptq_config: GptqConfig,
    revision: str,
    debug: bool,
    task_des: str,
    seed_path: str,
    out_path: str,
    start_idx: int,
    stride: int,
    task: str,
    sample_path: str = None,
    repeat: int = 1,
    samples: list = None,
    dump: bool= False,
):
    # Model
    print("Loading LLM ...")
    model, tokenizer = load_model(
        model_path,
        device,
        num_gpus,
        max_gpu_memory,
        load_8bit,
        cpu_offloading,
        gptq_config,
        revision,
        debug,
    )

    messages = []
    # load task_des
    if os.path.exists(task_des):
        lines = open(task_des, 'r', encoding='utf-8').readlines()
        messages.append("".join(lines[:-1]))
        messages.append(lines[-1])
    
    # load seed examples
    if task in ["gen_instruct", "multi_tgt"]:
        files = os.listdir(seed_path)
        seeds = [x for x in files if "des" not in x]
        for seed in seeds:
            messages.append(open(os.path.join(seed_path, seed), 'r').read())
            messages.append(open(os.path.join(seed_path, seed[:-4]+"_des.txt")).read())
    elif task in ["level_instruct"]:
        seed_data = open(seed_path, 'r', encoding='utf-8').readlines()
        seeds = [_.strip().split("|") for _ in seed_data]
        for seed in seeds:
            messages.extend(seed)
    
    # Chat
    def new_chat():
        if conv_template:
            conv = get_conv_template(conv_template)
        else:
            conv = get_conversation_template(model_path)
        return conv
    
    # load samples
    assert sample_path is not None or samples is not None, \
        logging.error("sample_path and samples can not both be None!")
    results = []
    if task == "gen_instruct":
        if samples is not None:
            sp_lines = samples
        else:
            sp_lines = open(sample_path, 'r', encoding='utf-8').readlines()
            sp_lines = [json.loads(line.strip()) for line in sp_lines]
        end_idx = min(start_idx+stride, len(sp_lines))
        
        if dump and out_path:
            fout = open(out_path, 'w', encoding='utf-8')

        for sp_line in tqdm(sp_lines[start_idx:end_idx]):
            image_id = sp_line["image_id"] if "image_id" in sp_line else sp_line["filename"]
            assert "content" in sp_line or "caption" in sp_line
            content_caption = (sp_line["content"] + "\n") if "content" in sp_line else (sp_line["caption"] + "\n")

            content_bbox = []
            for bbox in sp_line["bboxes"]:
                # {"bbox_id": 0, "bbox": [0.4, 0.06, 0.66, 0.78], "expressions": ["groom", "groom", "man"]}
                # expressions = random.sample(bbox["expressions"], min(len(bbox["expressions"]), 5))
                expressions = bbox["expressions"]
                bbox_uni = normbbox(bbox["bbox"], sp_line["width"], sp_line["height"])
                content_bbox.append(": ".join(["/".join(expressions), "["+", ".join([str(_) for _ in bbox_uni[0]])+"]"]))

            content_bbox = "\n".join(content_bbox)
            content = "\n".join([content_caption, content_bbox])

            result = copy.deepcopy(sp_line)
            result["llm_out"] = ''
            for _ in range(repeat):
                conv = new_chat()
                for i in range(len(messages)):
                    if i%2 == 0:
                        conv.append_message(conv.roles[0], messages[i])
                        conv.append_message(conv.roles[1], messages[i+1])
                conv.append_message(conv.roles[0], content)
                conv.append_message(conv.roles[1], None)
            
                generate_stream_func = generate_stream
                prompt = conv.get_prompt()

                gen_params = {
                    "model": model_path,
                    "prompt": prompt,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": max_new_tokens,
                    "stop": conv.stop_str,
                    "stop_token_ids": conv.stop_token_ids,
                    "echo": False,
                }
                
                # chatio.prompt_for_output(conv.roles[1])
                output_stream = generate_stream_func(model, tokenizer, gen_params, device)
                t = time.time()
                outputs = chatio.stream_output(output_stream)
                duration = time.time() - t
                conv.update_last_message(outputs.strip())
                
                if debug:
                    num_tokens = len(tokenizer.encode(outputs))
                    msg = {
                        "conv_template": conv.name,
                        "prompt": prompt,
                        "outputs": outputs,
                        "speed (token/s)": round(num_tokens / duration, 2),
                    }
                    logging.debug(f"\n{msg}\n")
                result["llm_out"] += outputs
            results.append(result)
            if dump and out_path:
                fout.write(json.dumps(result, ensure_ascii=False)+'\n')
                fout.flush()

    elif task == "level_instruct":
        if samples is not None:
            sp_lines = samples
        else:
            data = json.load(open(sample_path, 'r')) # refcoco format
            sp_lines = data["images"]

        if dump and out_path:
            fout = open(out_path, 'w', encoding="utf-8")

        end_idx = min(start_idx+stride, len(sp_lines))
        
        pattern = r": level ([0-3])"
        results = []
        for idx, sp_line in enumerate(tqdm(sp_lines[start_idx:end_idx])):
            result = copy.deepcopy(sp_line)
            if "expressions" in sp_line: # refcoco format
                result["levels"] = []
                for content in sp_line["expressions"]:
                    conv = new_chat()
                    for i in range(len(messages[:42])):
                        if i%2 == 0:
                            conv.append_message(conv.roles[0], "grade description: " + messages[i])
                            conv.append_message(conv.roles[1], f"My grading for description {messages[i]}: "+messages[i+1])
                    conv.append_message(conv.roles[0], "grade description: " + content)
                    conv.append_message(conv.roles[1], None)
                    
                    generate_stream_func = generate_stream
                    prompt = conv.get_prompt()

                    gen_params = {
                        "model": model_path,
                        "prompt": prompt,
                        "temperature": temperature,
                        "repetition_penalty": repetition_penalty,
                        "max_new_tokens": max_new_tokens,
                        "stop": conv.stop_str,
                        "stop_token_ids": conv.stop_token_ids,
                        "echo": False,
                    }

                    # chatio.prompt_for_output(conv.roles[1])
                    output_stream = generate_stream_func(model, tokenizer, gen_params, device)
                    t = time.time()
                    duration = time.time() - t
                    outputs = chatio.stream_output(output_stream)
                    conv.update_last_message(outputs.strip())
                    
                    match = re.findall(pattern, outputs, re.M)
                    if match and len(match)==1:
                        level = int(match[0])
                    else:
                        level = 1
                    # save level
                    result["levels"].append(level)

            elif "bboxes" in sp_line: # jsonline format
                for index, bbox in enumerate(sp_line["bboxes"]):
                    for exptype in ["expressions", "expressions_llm", "expressions_vlm"]:
                        if exptype in bbox:
                            result["bboxes"][index][f"{exptype}_levels"] = []
                            for content in bbox[exptype]:
                                conv = new_chat()
                                for i in range(len(messages[:42])):
                                    if i%2 == 0:
                                        conv.append_message(conv.roles[0], "grade description: " + messages[i])
                                        conv.append_message(conv.roles[1], f"My grading for description {messages[i]}: "+messages[i+1])
                                conv.append_message(conv.roles[0], "grade description: " + content)
                                conv.append_message(conv.roles[1], None)
                                
                                generate_stream_func = generate_stream
                                prompt = conv.get_prompt()

                                gen_params = {
                                    "model": model_path,
                                    "prompt": prompt,
                                    "temperature": temperature,
                                    "repetition_penalty": repetition_penalty,
                                    "max_new_tokens": max_new_tokens,
                                    "stop": conv.stop_str,
                                    "stop_token_ids": conv.stop_token_ids,
                                    "echo": False,
                                }

                                # chatio.prompt_for_output(conv.roles[1])
                                output_stream = generate_stream_func(model, tokenizer, gen_params, device)
                                t = time.time()
                                duration = time.time() - t
                                outputs = chatio.stream_output(output_stream)
                                conv.update_last_message(outputs.strip())
                                
                                match = re.findall(pattern, outputs, re.M)
                                if match and len(match)==1:
                                    level = int(match[0])
                                else:
                                    level = 1
                                # save level
                                result["bboxes"][index][f"{exptype}_levels"].append(level)

            if "clusters_intsc" in sp_line:
                for index, cluster in enumerate(sp_line["clusters_intsc"]):
                    assert "expressions" in cluster
                    result["clusters_intsc"][index]["expressions_levels"] = [5]*len(cluster["expressions"])

            if "clusters" in sp_line:
                for index, cluster in enumerate(sp_line["clusters"]):
                    for exptype in ["expressions_llm"]:
                        if exptype in cluster:
                            result["clusters"][index][f"{exptype}_levels"] = [5]*len(cluster[exptype])

            if "clusters_conj" in sp_line:
                for index, cluster in enumerate(sp_line["clusters_conj"]):
                    assert "expressions" in cluster
                    result["clusters_conj"][index]["expressions_levels"] = [4]*len(cluster["expressions"])

            results.append(result)
            if dump and out_path:
                fout.write(json.dumps(result, ensure_ascii=False)+'\n')
                fout.flush()
        
    elif task == "multi_tgt":
        def parse_cluster(output):
            ignore_exps = ["people in the image", "people in image", "left side", "right side", \
                           "left and right side", "right and left side", "in the middle", "left or right"]
            expressions = []
            pattern = r'''Summary of common properties of given objects:\n\#\# (.*)'''
            match = re.findall(pattern, output, re.M)
            if match and len(match)==1:
                expressions_ini = list(set((match[0]+" ").split("; ")[:-1]))
                for exp in expressions_ini:
                    isignore = False
                    for ignore_exp in ignore_exps:
                        if ignore_exp in exp:
                            isignore = True
                            break
                    if not isignore:
                        expressions.append(exp)
            return expressions
        
        def get_expall(bbox):
            expall = []
            for exptype in ["expressions", "expressions_vlm", "expressions_llm"]:
                if exptype in bbox:
                    expall += bbox[exptype]
            expall = list(set(expall))
            return expall

        if samples is not None:
            sp_lines = samples
        else:
            sp_lines = open(sample_path, 'r', encoding='utf-8').readlines()
            sp_lines = [json.loads(line.strip()) for line in sp_lines]
        end_idx = min(start_idx+stride, len(sp_lines))
        
        if dump and out_path:
            fout = open(out_path, 'w', encoding='utf-8')

        for sp_line in tqdm(sp_lines[start_idx:end_idx]):
            image_id = sp_line["image_id"] if "image_id" in sp_line else sp_line["filename"]
            assert "clusters" in sp_line, "ERROR: Not found clusters in meta_dict! \
                Please do clustering before generating expressions for clusters"
            clusters_intsc = sp_line["clusters_intsc"] if "clusters_intsc" in sp_line else []
            clusters_ana = sp_line["clusters"] if "clusters" in sp_line else []
            clusters_list = clusters_intsc + clusters_ana
            cluster_ids = []
            for cluster in clusters_list:
                if cluster["bbox_ids"] not in cluster_ids:
                    cluster_ids.append(cluster["bbox_ids"])
            
            bboxes = sp_line["bboxes"]
            result = copy.deepcopy(sp_line)
            result["clusters"] = []
            for cluster in cluster_ids:
                content = ["Objects and their descriptions:"]
                content += [f"## object {i}: " + ", ".join(get_expall(bboxes[i])) \
                            for i in cluster]
                content += ["Please find an summerize the similar properties of given objects."]
                content = "\n".join(content)

                conv = new_chat()
                for i in range(len(messages)):
                    if i%2 == 0:
                        conv.append_message(conv.roles[0], messages[i])
                        conv.append_message(conv.roles[1], messages[i+1])
                conv.append_message(conv.roles[0], content)
                conv.append_message(conv.roles[1], None)
                
                generate_stream_func = generate_stream
                prompt = conv.get_prompt()

                gen_params = {
                    "model": model_path,
                    "prompt": prompt,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": max_new_tokens,
                    "stop": conv.stop_str,
                    "stop_token_ids": conv.stop_token_ids,
                    "echo": False,
                }
                
                # chatio.prompt_for_output(conv.roles[1])
                output_stream = generate_stream_func(model, tokenizer, gen_params, device)
                outputs = chatio.stream_output(output_stream)
                conv.update_last_message(outputs.strip())
                
                outputs = parse_cluster(outputs)
                if len(outputs) > 0:
                    cluster_dic = dict(
                        bbox_ids=cluster,
                        expressions_llm=outputs
                    )
                    result["clusters"].append(cluster_dic)
            results.append(result)
            if dump and out_path:
                fout.write(json.dumps(outdata, ensure_ascii=False)+'\n')
                fout.flush()
    del model
    return results
