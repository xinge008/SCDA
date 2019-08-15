from __future__ import division

from .pycocotools.coco import COCO

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
from io import StringIO
from PIL import Image
import pickle as pk
import os
import json
import cv2
import logging
#logger = logging.getLogger('global')

class COCODataset(Dataset):
    category_to_class = {}
    class_to_category = {}
    person_keypoints = { 
            0:"Nose",
            1:"LEye", 2:"REye",
            3:"LEar", 4:"REar",
            5:"LShoulder", 6:"RShoulder",
            7:"LElbow", 8:"RElbow",
            9:"LWrist", 10:"RWrist",
            11:"LHip",  12:"RHip",
            13:"LKnee", 14:"RKnee",
            15:"LAnkle", 16:"RAnkle"}
    keypoint_lr_pairs = [[1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14], [15,16]]

    @staticmethod
    def get_class(category_id):
        return COCODataset.category_to_class[category_id]

    @staticmethod
    def get_category(class_id):
        return COCODataset.class_to_category[class_id]
    
    def __init__(self, root_dir, anno_file, transform_fn = None, normalize_fn=None,
        has_keypoint = False, 
        has_mask = False):

        self.root_dir = root_dir
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.has_keypoint = has_keypoint
        self.has_mask = has_mask

        #logger.info("building dataset from %s" % anno_file)
        self.coco = COCO(anno_file)
        category_ids = self.coco.cats.keys()

        # since coco 80 category ids is not contiguous
        COCODataset.category_to_class = {c:i+1 for i, c in enumerate(sorted(category_ids))}
        COCODataset.class_to_category = {i+1:c for i, c in enumerate(sorted(category_ids))}
        self.category_to_class = COCODataset.category_to_class
        self.class_to_category = COCODataset.class_to_category
        #logger.info('category_to_class:{}'.format(COCODataset.category_to_class))
        #logger.info('class_to_category:{}'.format(COCODataset.class_to_category))
        
        for img in self.coco.imgs.values():
            img['aspect_ratio'] = float(img['height']) / img['width']
        self.img_ids = list(set([_['image_id'] for _ in self.coco.anns.values()]))
        self.aspect_ratios = [self.coco.imgs[ix]['aspect_ratio'] for ix in self.img_ids]
        #logger.info('dataset scale:{}'.format(len(self.img_ids)))

    def __len__(self):
        return len(self.img_ids)

    def _fake_zero_data(self, *size):
        return np.zeros(size)

    def __getitem__(self, idx):
        '''
        Args: index of data
        Return: a single data:
            image_data: FloatTensor, shape [1, 3, h, w]
            image_info: list of [resized_image_h, resized_image_w, resize_scale, origin_image_h, origin_image_w]
            bboxes: np.array, shape [N, 5] (x1,y1,x2,y2,label)
            keypoints: np.array, shape [N, num_keypoints, 3] (x,y,v)
            masks: np.array, shape [N, h, w]
            filename: str
        Warning:
            we will feed fake ground truthes if None
        '''
        img_id = self.img_ids[idx]

        meta_img = self.coco.imgs[img_id]
        meta_annos = self.coco.imgToAnns[img_id]
        filename = os.path.join(self.root_dir, meta_img['file_name'])
        image_h, image_w = meta_img['height'], meta_img['width']

        num_annos = len(meta_annos)
        bboxes, ignore_regions, keypoints, masks = [], [], [], []
        for ann in meta_annos:
            if ann['iscrowd']:
                ignore_regions.append(ann['bbox'])
                continue
            #label = COCODataset.get_class(ann['category_id'])
            label = self.category_to_class[ann['category_id']]
            bbox = np.array(ann['bbox'] + [label])
            # ann['bbox'] store bbox as x1,y1,w,h
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            bboxes.append(bbox)

            if self.has_keypoint:
                kpt = np.array(ann['keypoints']).reshape(-1, 3)
                keypoints.append(kpt)
            else:
                keypoints.append(self._fake_zero_data(len(COCODataset.person_keypoints), 3))

            if self.has_mask:
                masks.append(self.coco.annToMask(ann))
            else:
                masks.append(self._fake_zero_data(image_h, image_w))
        if len(ignore_regions) == 0:
            ignore_regions.append(self._fake_zero_data(4))
        ignore_regions = np.array(ignore_regions, dtype = np.float32)
        bboxes = np.array(bboxes, dtype = np.float32)
        keypoints = np.array(keypoints, dtype = np.float32)
        masks = np.array(masks, dtype = np.uint8)

        img = Image.open(filename)
        if img.mode == 'L':
            img = img.convert('RGB')
        assert(img.size[0]==image_w and img.size[1]==image_h)
        ## transform
        if self.transform_fn:
            resize_scale, img, bboxes, ignore_regions, keypoints, masks = \
                self.transform_fn(img, bboxes, ignore_regions, keypoints, masks)
        else:
            resize_scale = 1
        new_image_w, new_image_h = img.size
        ## to tensor
        to_tensor = transforms.ToTensor()
        img = to_tensor(img)
        if self.normalize_fn != None:
            img = self.normalize_fn(img)
        return [img.unsqueeze(0),
                [new_image_h, new_image_w, resize_scale, image_h, image_w],
                bboxes,
                ignore_regions,
                keypoints,
                masks,
                filename]

class COCOTransform(object):
    def __init__(self, sizes, max_size, flip=False):
        if not isinstance(sizes, list):
            sizes = [sizes]
        self.scale_min = min(sizes)
        self.scale_max = max(sizes)
        self.max_size = max_size
        self.flip = flip

    def flip_keypoints(self, keypoints, image_w):
        '''
        Args:
            keypoints: 3*K elements, this argument is modified after this call.
            image_w: image witdh
        Return: flipped keypoints
        Side effect: keypints is modified
        '''
        if keypoints.size == 0:
            return keypoints
        N, K = keypoints.shape[:2]
        keypoints = keypoints.reshape(-1, 3)
        labeled = np.where(keypoints[:, 2] > 0)[0]
        keypoints[labeled, 0] = image_w - 1 - keypoints[labeled, 0]
        keypoints = keypoints.reshape(N, K, 3)
        for left, right in COCODataset.keypoint_lr_pairs:
            keypoints[:, left, :], keypoints[:, right, :] = \
                    keypoints[:, right, :], keypoints[:, left, :].copy()
        #logger.debug('lr_pairs:{}'.format(COCODataset.keypoint_lr_pairs))
        return keypoints


    def __call__(self, img, bboxes, ignore_regions, keypoints, masks):
        image_w, image_h = img.size
        short = min(image_w, image_h)
        large = max(image_w, image_h)

        size = np.random.randint(self.scale_min, self.scale_max + 1)
        scale = min(size / short, self.max_size / large)

        new_image_w, new_image_h = int(image_w * scale), int(image_h * scale)
        # test upper bound of gpu memory
        # new_image_w = new_image_h = self.max_size
        img = img.resize((new_image_w, new_image_h))
        if bboxes.shape[0] > 0:
            bboxes[:, :4] *= scale
        keypoints[keypoints > 0] *= scale
        # don't use cv2 since cv2.resize comsume gpu memopy in cluster16
        # masks = np.array([cv2.resize(m, img.size) for m in masks])
        if masks.size > 0:
            #masks = np.array(Image.fromarray(masks.tranpose(1,2,0)).resize(img.size))
            masks = np.array([np.array(Image.fromarray(m).resize(img.size)) for m in masks])
        ignore_regions[:, :4] *= scale

        if self.flip:
            if np.random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if bboxes.shape[0] > 0:
                    bboxes[:, 0], bboxes[:, 2] = \
                            new_image_w-1 - bboxes[:, 2], new_image_w-1 - bboxes[:, 0]
                masks = np.array([m[..., ::-1] for m in masks])
                keypoints = self.flip_keypoints(keypoints, new_image_w)
                ignore_regions[:, 0], ignore_regions[:,2] = \
                        new_image_w-1 - ignore_regions[:, 2], new_image_w-1 - ignore_regions[:, 0]
        return scale, img, bboxes, ignore_regions, keypoints, masks

