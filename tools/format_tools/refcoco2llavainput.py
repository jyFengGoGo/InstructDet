import os,json
from collections import defaultdict

def load_refcoco_bbox_dic(refcoco_label):
    bbox_dic = dict() 
    # filename: {width:, height:, 
    #            expressions:{bboxstr:[expressions]}, 
    #            anno:{bboxstr:{bbox:, segmentations:}}}
    data = json.load(open(refcoco_label, 'r', encoding='utf-8'))
    imgid2bboxes = defaultdict(list)
    for anno in data["annotations"]:
        imgid = anno["image_id"]
        imgid2bboxes[str(imgid)].append(anno)

    for exp in data["images"]:
        imgid = exp["id"]
        filename = exp["file_name"]
        width = exp["width"]
        height = exp["height"]
        expressions = exp["expressions"]
        anno_dics = imgid2bboxes[str(imgid)]
        bbox = [anno_dic["bbox"] for anno_dic in anno_dics]
        bboxstr = "_".join(sorted(["_".join([str(_) for _ in box]) for box in bbox]))
        
        if filename in bbox_dic:
            if bboxstr in bbox_dic[filename]["anno"]:
                bbox_dic[filename]["expressions"][bboxstr].extend(expressions)
            else:
                bbox_dic[filename]["expressions"][bboxstr] = expressions
                bbox_dic[filename]["anno"][bboxstr] = anno_dics
        else:
            bbox_dic[filename] = dict(
                width=width,
                height=height,
                expressions={bboxstr:expressions},
                anno={bboxstr:anno_dics}
            )
    return bbox_dic


def build4llava(bbox_dic):
    # {filename:xx.jpg, height:1, width:1, bboxes:[], hints:[]}
    llava_dics= []
    for filename in bbox_dic:
        temp_dic = dict(
            filename=filename,
            height=bbox_dic[filename]["height"],
            width=bbox_dic[filename]["width"],
            bboxes=[],
            hints=[],
        )
        for bboxid, (bboxstr, expressions) in enumerate(bbox_dic[filename]["expressions"].items()):
            anno_dics = bbox_dic[filename]["anno"][bboxstr]
            temp_dic["bboxes"].append(
                dict(
                    bbox_id=bboxid,
                    bbox=[anno_dic["bbox"] for anno_dic in anno_dics],
                    expressions=expressions,
                    bboxes_init_anno=anno_dics,
                )
            )
            temp_dic["hints"].append(expressions[0])
        llava_dics.append(temp_dic)
    return llava_dics


if __name__=="__main__":
    infile = "data/labels/refcoco/refcoco-unc/instances_testA.json"
    outfile = "data/labels/refcoco/refcoco-unc/llavainput_testA.json"
    llava_dics = build4llava(load_refcoco_bbox_dic(infile))
    with open(outfile, 'w') as fw:
        for llava_dic in llava_dics:
            fw.write(json.dumps(llava_dic, ensure_ascii=False)+'\n')
