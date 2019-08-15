from __future__ import division
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from io import StringIO
from PIL import Image
import pickle as pk
import os
import logging

def pil_loader(img_str):
    #buff = StringIO.StringIO()
    buff = StringIO()
    buff.write(img_str)
    buff.seek(0)
    with Image.open(buff) as img:
        return img.convert('RGB')
 
class ExampleDataset(Dataset):
    def __init__(self, root_dir, list_file, transform_fn, normalize_fn=None, memcached=False):
        #self.logger = logging.getLogger('global')
        self.root_dir = root_dir
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        # self.memcached = memcached

        #self.logger.info("building dataset from %s" % list_file)
        save_name = 'meta_%s'%(list_file.split('.')[0].strip('/').replace('/', '_'))
        ## load annotations if exist
        if os.path.exists(save_name):
            with open(save_name, 'rb') as f:
                self.metas = pk.load(f)
                self.num = len(self.metas)
                # aspect ratio of images for sampler sort
                self.aspect_ratios = [float(m[1])/m[2] for m in self.metas]
            return
        ## otherwise parse annotations
        with open(list_file) as f:
            lines = f.readlines()
        self.metas = []
        count = 0
        i = 0
        while i < len(lines):
            img_ig = []
            img_gt = []
            labels = []
            img_name = lines[i + 1].rstrip()
            img_height = float(lines[i + 3])
            img_width = float(lines[i + 4])
            img_ig_size = int (lines[i + 6])
            i += 7
            for j in range(img_ig_size):
                sp = lines[i + j].split()
                img_ig.append([float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3])])
            if len(img_ig) == 0:
                img_ig.append([0,0,0,0])
            i += img_ig_size
            img_gt_size = int(lines[i])
            i += 1
            for j in range(img_gt_size):
                sp = lines[i + j].split()
                img_gt.append([float(sp[1]),float(sp[2]),float(sp[3]),float(sp[4])])
                labels.append(int(sp[0]))
            i += img_gt_size
            count += 1
            #if count % 100 == 0:
            #    self.logger.info(count)
            self.metas.append([img_name, img_height, img_width, np.array(img_gt), np.array(labels), np.array(img_ig)])
        with open(save_name, 'wb') as f:
            pk.dump(self.metas, f)
        #self.logger.info("read meta done")
        self.num = len(self.metas)
        # aspect ratio of images for sampler sort
        self.aspect_ratios = [float(m[1])/m[2] for m in self.metas]
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.metas[idx][0])
        h, w, bbox, labels, ignores = self.metas[idx][1:]
        bbox = bbox.astype(np.float32)
        ignores = ignores.astype(np.float32)
        labels = labels.astype(np.float32)
        img = Image.open(filename)
        if img.mode == 'L':
            img = img.convert('RGB')
        assert(img.size[0]==w and img.size[1]==h)
        ## det transform
        img, bbox, resize_scale, ignores = self.transform_fn(img, bbox, ignores)
        new_w, new_h = img.size
        ## to tensor
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        if self.normalize_fn != None:
            img = self.normalize_fn(img)
        bbox = np.hstack([bbox, labels[:, np.newaxis]])
        return [img.unsqueeze(0),
                torch.Tensor([new_h, new_w, resize_scale]),
                torch.from_numpy(bbox),
                torch.from_numpy(ignores),
                filename]


class ExampleTransform(object):
    def __init__(self, sizes, max_size, flip=False):
        if not isinstance(sizes, list):
            sizes = [sizes]
        self.scale_min = min(sizes)
        self.scale_max = max(sizes)
        self.max_size = max_size
        self.flip = flip

    def __call__(self, img, bbox, ignores):

        w, h = img.size
        short = min(w, h)
        large = max(w, h)

        size = np.random.randint(self.scale_min, self.scale_max + 1)
        scale = min(size / short, self.max_size / large)
        new_w, new_h = int(w * scale), int(h * scale)

        new_img = img.resize((new_w, new_h))

        new_bbox = np.array(bbox)
        new_bbox[:, 0] = np.floor(new_bbox[:, 0] * scale)
        new_bbox[:, 1] = np.floor(new_bbox[:, 1] * scale)
        new_bbox[:, 2] = np.ceil(new_bbox[:, 2] * scale)
        new_bbox[:, 3] = np.ceil(new_bbox[:, 3] * scale)
        new_ignores = np.array(ignores)
        if new_ignores.shape[0] > 0:
            new_ignores[:, 0] = np.floor(new_ignores[:, 0] * scale)
            new_ignores[:, 1] = np.floor(new_ignores[:, 1] * scale)
            new_ignores[:, 2] = np.ceil(new_ignores[:, 2] * scale)
            new_ignores[:, 3] = np.ceil(new_ignores[:, 3] * scale)

        if self.flip:
            if np.random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_bbox[:, 0], new_bbox[:, 2] = new_w - new_bbox[:, 2], new_w - new_bbox[:, 0]
                if new_ignores.shape[0] > 0:
                    new_ignores[:, 0], new_ignores[:,2] = new_w - new_ignores[:, 2], new_w - new_ignores[:, 0]
        return new_img, new_bbox, scale, new_ignores
