import os
from PIL import Image
from PIL import ImageDraw

import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class REFCOCODataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode("pilrgb", handler=wds.warn_and_continue),
            wds.to_tuple("jpg", "json", handler=wds.warn_and_continue),
            wds.map_tuple(self.vis_processor, handler=wds.warn_and_continue),
            wds.map(self.to_dict, handler=wds.warn_and_continue),
        )

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }


class REFCOCOAlignDataset(CaptionDataset):

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]                      #用序号将这一条训练样本索引出来

        img_file = '{}.jpg'.format(ann["image_id"])       #找到训练样本对应的图像名称
        image_path = os.path.join(self.vis_root, img_file)   #找到训练样本对应的图像路径
        image = Image.open(image_path).convert("RGB")     #找到这个图像

        #给图像画上红色的检测框
        bbox = ann["bbox"]
        draw = ImageDraw.Draw(image)
        length = min(bbox[2], bbox[3])
        if length >= 100 and length < 200:
            line_width = 5
        elif length >= 50 and length < 100:
            line_width = 4
        elif length >= 20 and length < 50:
            line_width = 2
        elif length < 20:
            line_width = 1
        else:
            line_width = 6

        draw.rectangle([bbox[0],bbox[1],bbox[0] + bbox[2],bbox[1] + bbox[3]], outline=(255,0,0), width=line_width) # [左上角x，左上角y，右下角x，右下角y]，outline边框颜色
 
        image = self.vis_processor(image)                #处理一下这个图像
        caption = ann["caption"]                         #图像对应的文本

        for i in range(len(bbox)):
            bbox[i] = str(bbox[i])
        bbox = ",".join(bbox)
        # 返回图像，文本和图像id (根据annotions顺序的)
        return {
            "image": image,
            "text_input": caption,
            "image_id": ann["image_id"],
            "bbox": bbox,
            "raw_id": ann["raw_id"],
            "id": ann["id"]
        }