import os
import json
from easydict import EasyDict
import yaml
import argparse

from datasets.dataset import ExpressionDataset
from modules.fastchat.fastchat.serve.ginstruct import main as generate_expression
from modules.instruct_filter.clip_filter import CLIPFilter
from tools.utils import expand_expression, cluster_by_intersection, conjunction_expand

from datetime import datetime
import logging
logging.basicConfig(filename=datetime.now().strftime("logs/%Y-%m-%d-%H-%M-%S.log"), \
                    filemode="w", \
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", \
                    datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = EasyDict(config)
    return config


def update_lmm_config(cfg):
    cfg.llava_config.meta_file = cfg.meta_file
    cfg.llava_config.image_prefix = cfg.image_prefix
    cfg.llava_config.outpath = cfg.outpath
    cfg.llava_config.startidx = cfg.startidx
    cfg.llava_config.stride = cfg.stride
    cfg.llava_config.petrel_conf = cfg.petrel_conf


def update_llm_config(cfg, task=None):
    assert task in ["gen_instruct", "multi_tgt", "level_instruct"]
    cfg.llama_config.task = task
    task_cfg = getattr(cfg.llama_config.tasks, task)
    cfg.llama_config.task_des = task_cfg.task_des
    cfg.llama_config.seed_path = task_cfg.seed_path
    cfg.llama_config.sample_path = getattr(task_cfg, "sample_path", None)
    cfg.llama_config.temperature = getattr(task_cfg, "temperature", 0.7)
    cfg.llama_config.repeat = getattr(task_cfg, "repeat", 1)
    cfg.llama_config.out_path = cfg.outpath
    cfg.llama_config.start_idx = 0
    cfg.llama_config.stride = 1e+10


def load_dataset(cfg):
    dataset = ExpressionDataset(cfg.image_prefix, cfg.meta_file, petrel_conf=cfg.petrel_conf)
    jsonlines = dataset.metas[cfg.startidx:min(len(dataset), cfg.startidx+cfg.stride)]
    return jsonlines


def main(cfg):
    exptypes = []
    ### global prompt pipeline
    if cfg.use_global_ppl:
        if not cfg.with_caption:
            logging.info("Generating global image caption using LLaVA...")
            from modules.llava.llava.eval.run_llava_dsp import describe_image
            update_lmm_config(cfg)
            jsonlines = describe_image(cfg.llava_config)
        else:
            jsonlines = load_dataset(cfg)

        logging.info("Generating instructions from image caption by LLM...")
        update_llm_config(cfg, task="gen_instruct")
        jsonlines = generate_expression(cfg.llama_config, samples=jsonlines)

        logging.info("Parsing LLM outputs into instructions...")
        jsonlines = [expand_expression(line)[0] for line in jsonlines]
        exptypes.append("expressions_llm")

    ### local prompt pipeline
    if cfg.use_local_ppl:
        if not cfg.use_global_ppl:
            jsonlines = load_dataset(cfg)
        
        logging.info("Generating local caption by VLM...")
        single_only = cfg.local_ppl_config.single_only
        vlm_config = cfg.local_ppl_config
        if vlm_config.type == "minigpt4":
            from modules.minigpt4.generate import MiniGPT4Generator
            vlm_config.minigpt4.petrel_conf = cfg.petrel_conf
            vlm_config.minigpt4.single_only = single_only
            generator = MiniGPT4Generator(vlm_config.minigpt4)
            jsonlines = generator(jsonlines)
            exptypes.append("expressions_vlm")
            if not single_only:
                exptypes.append("expressions_single")
            del generator
        else:
            logging.error(f"VLM method {vlm_config.type} is NotImplemented.")
    
    ### instruction filtering
    logging.info("Filtering generated instructions...")
    # 1. VL matchness filtering
    filter_config = cfg.filter_config
    if filter_config.type == "CLIP":
        filter_config.CLIP.petrel_conf = cfg.petrel_conf
        filter = CLIPFilter(filter_config.CLIP)
        jsonlines = filter(jsonlines, exptypes=exptypes)
    else:
        logging.error(f"Filtering method {filter_config.type} is NotImplemented.")
    # 2. TODO: L exclusive filtering, only for exp. generated for object in initial clusters
    if cfg.use_local_ppl and not cfg.local_ppl_config.single_only:
        pass
    
    ### multi-object instruction
    if cfg.find_inter:
        logging.info("Start finding instruction intersections of bboxes.")
        jsonlines = cluster_by_intersection(jsonlines)

    if cfg.use_cluster:
        logging.info("Start clustering bboxes according to instructions.")
        cluster_config = cfg.multi_object.cluster
        if cluster_config.type == "DBSCAN":
            from modules.text_cluster.bert_dbscan import expression_clustering
            # 1. find potential multi-object instructions by DBSCAN
            jsonlines = expression_clustering(cluster_config.DBSCAN, samples=jsonlines)
            # 2. generate instructions for common attributes by LLM
            update_llm_config(cfg, task="multi_tgt")
            jsonlines = generate_expression(cfg.llama_config, samples=jsonlines)
            # 3. cluster instruction filtering
            jsonlines = filter.cluster_filter(jsonlines, exptypes=["expressions_llm"])
        else:
            logging.error(f"Clustering method {cluster_config.type} is NotImplemented.")
            
    if cfg.use_conjunction:
        logging.info("Combining instructions by conjunctions: ',' or 'and'.")
        conj_config = cfg.multi_object.conjunction
        jsonlines = conjunction_expand(jsonlines, conj_config)

    ### instruction leveling/grouping
    if cfg.use_group:
        logging.info("Grouping instructions by LLM...")
        update_llm_config(cfg, task="level_instruct")
        jsonlines = generate_expression(cfg.llama_config, samples=jsonlines)

    # dump result
    if cfg.outpath:
        outpath = cfg.outpath
    else:
        outpath = cfg.out_path
    if isinstance(outpath, list):
        logging.info(f"Saving results(jsonlines) into {outpath[0]}")
        outpath = outpath[0]
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    with open(os.path.join(outpath, "expressions.jsonl"), 'w') as fout:
        for line in jsonlines:
            del line["image"]
            logging.debug(json.dumps(line, ensure_ascii=False, indent=2))
            fout.write(json.dumps(line, ensure_ascii=False)+'\n')


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Expression Generation")
    parser.add_argument("-c", "--config-path", type=str, default="configs/instructdet.yaml")
    parser.add_argument("-o", "--out-path", type=str, default="outputs/")
    parser.add_argument("--startidx", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    if not cfg.outpath:
        cfg.out_path = args.out_path
    cfg.startidx = args.startidx
    cfg.stride = args.stride

    logging.info(cfg)
    main(cfg)
