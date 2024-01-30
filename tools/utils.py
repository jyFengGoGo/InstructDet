import re
import random
from tqdm import tqdm
import copy
from collections import Counter, defaultdict
from itertools import combinations

def expand_expression(jsonline_wt_llmout, uncased=False):
    """
        parse large language model outputs into expressions for corresponding bboxes,
        update each bbox_info_dict in jsonline['bboxes'] with extra key='expressions_llm'
    """
    ignore_exps = ["person in the image", "people in the image", \
                   "item in the image", "items in the image", "item in the scene", \
                   "object in the image", "objects in the image", "object in the scene", \
                   "focus of the image", "focus of the scene", "main object of image", \
                   "body part"]
    
    all_catg = []
    all_catg2boxidx = dict()
    for idx, bbox in enumerate(jsonline_wt_llmout["bboxes"]):
        categ = "/".join(sorted(bbox["expressions"]))
        all_catg.append(categ)
        all_catg2boxidx[categ] = idx
        bbox["expressions_llm"] = set()

    exp_num = 0
    pattern = r'''\[(.*)\]\n\(1\) (.*)\n\(2\) (.*)'''
    llm_out = jsonline_wt_llmout["llm_out"]
    match = re.findall(pattern, llm_out, re.M)
    if match:
        for (category, dsp1, dsp2) in match:
            if uncased:
                category = category.lower()
            sort_category = "/".join(sorted(category.split("/")))
            category_ls = set(category.split("/"))
            new_exps = dsp1.split(", ") + dsp2.split(", ")
            new_exps = [_ for _ in new_exps if len(_.split(" "))>1 and _ not in ignore_exps]
            if all_catg.count(sort_category) == 1: # category whole match
                match_boxid = all_catg2boxidx[sort_category]
                origin_exp_num = len(jsonline_wt_llmout["bboxes"][match_boxid]["expressions"])
                jsonline_wt_llmout["bboxes"][match_boxid]["expressions_llm"].update(new_exps)
                exp_num += len(jsonline_wt_llmout["bboxes"][match_boxid]["expressions_llm"]) - origin_exp_num
            elif all_catg.count(sort_category) == 0:
                for categ in all_catg:
                    categ_ls = set(categ.split("/"))
                    if categ_ls.intersection(category_ls):  # category sub match
                        match_boxid = all_catg2boxidx[categ]
                        origin_exp_num = len(jsonline_wt_llmout["bboxes"][match_boxid]["expressions"])
                        jsonline_wt_llmout["bboxes"][match_boxid]["expressions_llm"].update(new_exps)
                        exp_num += len(jsonline_wt_llmout["bboxes"][match_boxid]["expressions_llm"]) - origin_exp_num
                        break
    for bbox in jsonline_wt_llmout["bboxes"]:
        bbox["expressions_llm"] = list(bbox["expressions_llm"])
    return jsonline_wt_llmout, exp_num


def cluster_by_intersection(jsonlines):
    """
        find same expressions for different bboxes,
        remove the same expressions in each bbox, 
        and update jsonline with extra key='clusters_intsc'
    """
    def get_expall(bbox):
        expall = []
        for exptype in ["expressions", "expressions_vlm", "expressions_llm"]:
            if exptype in bbox:
                expall += bbox[exptype]
        expall = list(set(expall))
        return expall
        
    new_jsonlines = []
    for line in tqdm(jsonlines):
        bboxes = line["bboxes"]
        expall = []
        explist = []
        groups = defaultdict(list)
        for bbox in bboxes:
            bbox_expall = get_expall(bbox)
            expall += bbox_expall
            explist.append(bbox_expall)
        
        # find same instructions
        expcount = sorted(list(Counter(expall).items()), key=lambda x:x[1], reverse=True)
        for pair in expcount:
            if pair[1] > 1:
                exp = pair[0]
                group_ids = []
                for idx,exps in enumerate(explist):
                    if exp in exps:
                        group_ids.append(idx)
                groups["_".join([str(_) for _ in group_ids])].append(exp)
            else:
                break
        
        # updata cluster
        line["clusters_intsc"] = []
        if len(groups) > 0:
            for k,v in groups.items():
                bbox_ids = [int(_) for _ in k.split("_")]
                clusterdic = dict(
                    bbox_ids=bbox_ids,
                    expressions=v
                )
                line["clusters_intsc"].append(clusterdic)
                # remove instruction from single object
                for bbox_id in bbox_ids:
                    bbox_info = line["bboxes"][bbox_id]
                    for exp in v:
                        for exptype in ["expressions","expressions_vlm","expressions_llm"]:
                            if exptype in bbox_info and exp in bbox_info[exptype]:
                                bbox_info[exptype].remove(exp)
        new_jsonlines.append(line)
    return new_jsonlines


def conjunction_expand(jsonlines, cfg):
    boxnum_i, boxnum_j = cfg.num_scale
    new_jsonlines = []
    for line in tqdm(jsonlines):
        bboxes = line["bboxes"]
        bbox_ids = [i for i in range(len(bboxes))]

        newbboxes = copy.deepcopy(bboxes)
        for i, box in enumerate(bboxes):
            if len(box["bbox"]) > 1:
                bbox_ids.remove(i)
                continue
            exps = box["expressions"]
            for exp in exps:
                if " and " in exp or ", " in exp or len(exp.split(" "))>12 or len(exp.split(" "))<2:
                    newbboxes[i]["expressions"].remove(exp)
            if len(newbboxes[i]["expressions"]) == 0:
                bbox_ids.remove(i)
        
        bbox_num = len(bbox_ids)

        # gen combinations
        combins = []
        for clstr_bbox_num in range(boxnum_i, boxnum_j+1):
            if bbox_num < clstr_bbox_num:
                break
            combins += [c for c in combinations(bbox_ids, clstr_bbox_num)]
        sp_clstrs = random.sample(combins, min(len(combins), 2))
        line["clusters_conj"] = []
        for sp_clstr in sp_clstrs:
            clstr_exps = []
            for _ in range(2):
                exp = [random.choice(newbboxes[idx]["expressions"]) for idx in sp_clstr]
                random.shuffle(exp)
                if random.random() < 0.5:
                    clstr_exps.append(" and ".join(exp))
                else:
                    clstr_exps.append(", ".join(exp))
            clstr_dic = dict(
                bbox_ids=list(sp_clstr),
                expressions=list(set(clstr_exps))
            )
            line["clusters_conj"].append(clstr_dic)
        new_jsonlines.append(line)
    return new_jsonlines


def combine_exptypes(jsonlines, exptypes=[]):
    new_jsonlines = []
    for line in jsonlines:
        for bbox in line["bboxes"]:
            bbox["expressions_ori"] = bbox["expressions"]
            bbox_expall = []
            for exptype in exptypes:
                if exptype in bbox:
                    bbox_expall += bbox[exptype]
            bbox_expall = list(set(bbox_expall))
            bbox["expressions"] = bbox_expall
        new_jsonlines.append(line)
    return new_jsonlines
