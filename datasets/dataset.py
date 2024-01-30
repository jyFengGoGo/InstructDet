import os, json
import numpy as np
import random
import torch
from torch.utils.data import Dataset
try:
    import mc
except ImportError:
    pass
from petrel_client.client import Client
from .image_reader import build_image_reader

from collections import defaultdict


class BaseDataset(Dataset):
    def __init__(self,
                 root_dir,
                 meta_file=None,
                 transform=None,
                 read_from='mc',
                 petrel_conf='~/petreloss.conf',
                ):

        super(BaseDataset, self).__init__()

        self.root_dir = root_dir
        self.meta_file = meta_file
        self.transform = transform
        self.read_from = read_from
        self.initialized = False
        self.petrel_conf = petrel_conf

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

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/cache/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/cache/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def _init_petrel(self):
        if not self.initialized:
            self.client = Client(conf_path=self.petrel_conf)
            self.initialized = True

    def _init_osg(self):
        if not self.initialized:
            from spring_sdk import OSG
            self.osg_client = OSG(self.osg_server, secure=False)
            self.initialized = True

    def read_file(self, meta_dict):
        if self.read_from == 'fake':
            if self.initialized:
                filebytes = self.saved_filebytes
            else:
                filebytes = self.saved_filebytes = np.fromfile(meta_dict['filename'], dtype=np.uint8)
                self.initialized = True
        elif self.read_from == 'mc':
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(meta_dict['filename'], value)
            value_str = mc.ConvertBuffer(value)
            filebytes = np.frombuffer(value_str.tobytes(), dtype=np.uint8)
        elif self.read_from == 'petrel':
            self._init_petrel()
            value = self.client.get(meta_dict['filename'], update_cache=True)
            value = memoryview(value)
            filebytes = np.frombuffer(value, dtype=np.uint8)
        elif self.read_from == 'fs':
            filebytes = np.fromfile(meta_dict['filename'], dtype=np.uint8)
        elif self.read_from == 'osg':
            self._init_osg()
            img_str = self.osg_client.get_object(meta_dict['bucket'], meta_dict['key'])
            filebytes = np.fromstring(img_str, np.uint8)
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


class ExpressionDataset(BaseDataset):
    """
    Dataset to generate expressions using instructdet pipeline.

    Arguments:
        - root_dir (:obj:`str`): root directory of images
        - meta_file (:obj:`str`): name of meta file, each line is a json format info dict
        - transform (list of ``Transform`` objects): list of transforms
        - read_from (:obj:`str`): read type from the original meta_file
        - image_reader_type (:obj:`str`): reader type 'pil' or 'ks'

    Metafile example::
        "{"filename": "n01440764/n01440764_10026.JPEG",
          "label": 0}"
    """
    def __init__(self,
                 root_dir,
                 meta_file,
                 transform=None,
                 read_from='mc',
                 petrel_conf='~/petreloss.conf',
                ):
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
            self.num += len(lines)
            for line in lines:
                info = json.loads(line)
                info['filename'] = os.path.join(self.root_dir[i], info['filename'])
                self.metas.append(info)

        super(ExpressionDataset, self).__init__(root_dir=self.root_dir, meta_file=self.meta_file,
                                                read_from=read_from, transform=transform, petrel_conf=petrel_conf)

    def __len__(self):
        return self.num
