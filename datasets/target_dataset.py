# -*- coding: utf-8 -*-
# @Time    : 18-5-3 4:40
# @Author  : Xinge

from __future__ import division
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from io import StringIO
from PIL import Image
import pickle as pk
import os

def pil_loader(img_str):
    #buff = StringIO.StringIO()
    buff = StringIO()
    buff.write(img_str)
    buff.seek(0)
    with Image.open(buff) as img:
        return img.convert('RGB')

class TargetDataset(Dataset):
    def __init__(self, root_dir, list_file, normalize_fn=None, memcached=False, new_w=1024, new_h=512):
        # self.logger = logging.getLogger('global')
        self.root_dir = root_dir
        # self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.new_w = new_w
        self.new_h = new_h
        # self.memcached = memcached
        with open(list_file) as f:
            lines = f.readlines()
        self.metas = [x.strip() for x in lines]

        self.num = len(self.metas)
        # # aspect ratio of images for sampler sort
        # self.aspect_ratios = [float(m[1]) / m[2] for m in self.metas]

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.metas[idx])
        # h, w, bbox, labels, ignores = self.metas[idx][1:]
        # bbox = bbox.astype(np.float32)
        # ignores = ignores.astype(np.float32)
        # labels = labels.astype(np.float32)
        img = Image.open(filename)
        if img.mode == 'L':
            img = img.convert('RGB')
        # assert (img.size[0] == w and img.size[1] == h)
        ## det transform
        img = self.transform(img, self.new_w, self.new_h)
        # new_w, new_h = img.size
        ## to tensor
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        if self.normalize_fn != None:
            img = self.normalize_fn(img)
        # bbox = np.hstack([bbox, labels[:, np.newaxis]])
        return img


    def transform(self, img, new_w, new_h):
        """transform

        :param img:
        :param lbl:
        """
        new_img = img.resize((new_w, new_h))
        return new_img