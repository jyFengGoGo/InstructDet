import os
import json
from tqdm import tqdm

def jsonline2refcoco(infile, outfile):
    def get_expall(bbox):
        exp_all = []
        levels = []
        for exptype in ["expressions", "expressions_vlm", "expressions_llm"]:
            if exptype in bbox:
                for idx, exp in enumerate(bbox[exptype]):
                    if exp not in exp_all:
                        exp_all.append(exp)
                        if f"{exptype}_levels" in bbox:
                            levels.append(bbox[f"{exptype}_levels"][idx])
        return exp_all, None if len(levels) == 0 else levels

    cluster2exptype = {
        "clusters_intsc": "expressions",
        "clusters": "expressions_llm",
        "clusters_conj": "expressions"
    }

    refcoco = dict(
        images=[],
        annotations=[],
        categories=[{'supercategory': 'object', 'id': 1, 'name': 'object'}]
    )

    with open(infile, 'r', encoding='utf-8') as fin:
        jsonlines = fin.readlines()
    
    refpair_id = 0
    bbox_id = 0
    for jsonline in tqdm(jsonlines):
        jsonline = json.loads(jsonline.strip())
        for bbox in jsonline["bboxes"]:
            expressions, levels = get_expall(bbox)
            image_dic = dict(
                file_name=jsonline["filename"],
                height=jsonline['height'],
                width=jsonline['width'],
                id=refpair_id,
                expressions=expressions
            )
            if levels is not None:
                image_dic["levels"] = levels
            refcoco["images"].append(image_dic)
            for bbox_coor in bbox["bbox"]:
                anno_dic = dict(
                    bbox=bbox_coor, # xywh
                    image_id=refpair_id,
                    category_id=1,
                    iscrowd=0,
                    id=bbox_id
                )
                refcoco["annotations"].append(anno_dic)
                bbox_id += 1
            refpair_id += 1

        for clustertype in ["clusters", "clusters_conj"]:
            if clustertype in jsonline:
                for cluster in jsonline[clustertype]:
                    bbox_ids = cluster["bbox_ids"]
                    cluster_exptype = cluster2exptype[clustertype]
                    assert cluster_exptype in cluster
                    image_dic = dict(
                        file_name=jsonline["filename"],
                        height=jsonline['height'],
                        width=jsonline['width'],
                        id=refpair_id,
                        expressions=cluster[cluster_exptype]
                    )
                    if f"{cluster_exptype}_levels" in cluster:
                        image_dic["levels"] = cluster[f"{cluster_exptype}_levels"]
                    refcoco["images"].append(image_dic)
                    for bbox_id in bbox_ids:
                        for bbox_coor in jsonline["bboxes"][bbox_id]["bbox"]:
                            anno_dic = dict(
                                bbox=bbox_coor, # xywh
                                image_id=refpair_id,
                                category_id=1,
                                iscrowd=0,
                                id=bbox_id
                            )
                            refcoco["annotations"].append(anno_dic)
                            bbox_id += 1
                    refpair_id += 1
    
    with open(outfile, 'w') as fout:
        json.dump(refcoco, fout, ensure_ascii=False, indent=2)


if __name__=="__main__":
    infile = "outputs/expressions.jsonl"
    outfile = "outputs/expressions2refcoco.jsonl"
    jsonline2refcoco(infile, outfile)
