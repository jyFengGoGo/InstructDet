import os,json
from petrel_client.client import Client
from petrelbox.io import PetrelHelper
from pycocotools.coco import COCO
helper = PetrelHelper(conf_path="~/petreloss.conf")
import time
from collections import defaultdict
from tqdm import tqdm

class COCOwraper(COCO):
    def __init__(self, annotation_file=None):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            file_bytes = PetrelHelper.get_bytes(annotation_file)
            dataset = json.loads(file_bytes)
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset
            self.createIndex()

def load_refcoco_bbox_dic(file_name):
    bbox_dic = dict() 
    # filename: {width:, height:, 
    #            expressions:{bboxstr:[expressions]}, 
    #            anno:{bboxstr:{bbox:, segmentations:}}}
    # data = json.load(open(refcoco_label, 'r', encoding='utf-8'))
    coco = COCOwraper(file_name)
    img_ids = coco.getImgIds()
    
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs([img_id])[0]
        filename = img_info["file_name"]

        # test = helper.get_bytes("sdc:s3://MultiModal/Objects365/val/" + filename)
        # print(test)
        # exit()
        width = img_info["width"]
        height = img_info["height"]

        anns = coco.loadAnns(coco.getAnnIds([img_id]))

        for ann in anns:
            bbox = ann["bbox"]
            cat_id = ann['category_id']
            expressions = coco.loadCats([cat_id])[0]["name"]

            if not isinstance(bbox[0], list):
                bboxstr = "_".join([str(_) for _ in bbox])
            else:
                bboxstr = "_".join(sorted(["_".join([str(_) for _ in box]) for box in bbox]))
    
            if filename in bbox_dic:
                if bboxstr in bbox_dic[filename]["anno"]:
                    bbox_dic[filename]["expressions"][bboxstr].extend(expressions)
                else:
                    bbox_dic[filename]["expressions"][bboxstr] = expressions
                    bbox_dic[filename]["anno"][bboxstr] = ann
            else:
                bbox_dic[filename] = dict(
                    width=width,
                    height=height,
                    expressions={bboxstr:expressions},
                    anno={bboxstr:ann}
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
            anno_dic = bbox_dic[filename]["anno"][bboxstr]
            temp_dic["bboxes"].append(
                dict(
                    bbox_id=bboxid,
                    bbox=anno_dic["bbox"],
                    expressions=expressions,
                    image_id=anno_dic["image_id"],
                    bbox_init_id=anno_dic["id"]
                )
            )
            temp_dic["hints"].append(expressions[0])
        llava_dics.append(temp_dic)
    return llava_dics


if __name__=="__main__":
    infile = "data/labels/objects365/objects365_val.json"
    outfile = "data/labels/objects365/llavainput_val.json""
    llava_dics = build4llava(load_refcoco_bbox_dic(infile))
    with open(outfile, 'w') as fw:
        for llava_dic in llava_dics:
            fw.write(json.dumps(llava_dic, ensure_ascii=False)+'\n')
