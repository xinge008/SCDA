#encoding: utf-8

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import logging
#logger = logging.getLogger('global')

def to_np_array(x):
    if x is None:
        return None
    if isinstance(x, Variable): x = x.data
    return x.cpu().numpy() if torch.is_tensor(x) else np.array(x)

class COCODataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False):
        super(COCODataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                        num_workers, self._collate_fn, pin_memory, drop_last)
    def _collate_fn(self, batch):
        '''
        Return: a mini-batch of data:
            image_data: FloatTensor of image, with shape of [b, 3, max_h, max_w]
            image_info: np.array of shape [b, 5], (resized_image_h, resized_image_w, resize_scale, origin_image_h, origin_image_w)
            bboxes: np.array of shape [b, max_num_gts, 5]
            keypoints: np.array of shape[b, max_num_gts, k, 2]
            masks: np.array of shape [b, max_num_gts, max_h, max_w]
            filename: list of str
        '''
        batch_size = len(batch)

        zip_batch = list(zip(*batch))
        images = zip_batch[0]
        unpad_image_sizes = zip_batch[1]
        ground_truth_bboxes = zip_batch[2]
        ignore_regions = zip_batch[3]
        ground_truth_keypoints = zip_batch[4]
        ground_truth_masks = zip_batch[5]
        filenames = zip_batch[6]
        has_keyp = ground_truth_keypoints[0] is not None
        has_mask = ground_truth_masks[0] is not None


        max_img_h = max([_.shape[-2] for _ in images])
        max_img_w = max([_.shape[-1] for _ in images])

        max_img_h = int(np.ceil(max_img_h / 128.0) * 128)
        max_img_w = int(np.ceil(max_img_w / 128.0) * 128)

        max_num_gt_bboxes = max([_.shape[0] for _ in ground_truth_bboxes])
        max_num_ig_bboxes = max([_.shape[0] for _ in ignore_regions])
        assert(max_num_gt_bboxes > 0)
        assert(max_num_ig_bboxes > 0)

        padded_images = []
        padded_gt_bboxes = []
        padded_ig_bboxes = []
        padded_gt_keypoints = [] if has_keyp else None
        padded_gt_masks = [] if has_mask else None
        for b_ix in range(batch_size):
            img = images[b_ix]

            # pad zeros to right bottom of each image
            pad_size = (0, max_img_w - img.shape[-1], 0, max_img_h - img.shape[-2])
            padded_images.append(F.pad(img, pad_size, 'constant', 0).data.cpu())

            # pad zeros to gt_bboxes
            gt_bboxes = to_np_array(ground_truth_bboxes[b_ix])
            new_gt_bboxes = np.zeros([max_num_gt_bboxes, gt_bboxes.shape[-1]])
            new_gt_bboxes[range(gt_bboxes.shape[0]), :] = gt_bboxes
            padded_gt_bboxes.append(new_gt_bboxes)

            # pad zeros to ig_bboxes
            ig_bboxes = to_np_array(ignore_regions[b_ix])
            new_ig_bboxes = np.zeros([max_num_ig_bboxes, ig_bboxes.shape[-1]])
            new_ig_bboxes[range(ig_bboxes.shape[0]), :] = ig_bboxes
            padded_ig_bboxes.append(new_ig_bboxes)

            # pad zero to keypoints
            if has_keyp:
                keypoints = to_np_array(ground_truth_keypoints[b_ix])
                shape = keypoints.shape
                new_keypoints = np.zeros([max_num_gt_bboxes, shape[1], shape[2]])
                new_keypoints[range(keypoints.shape[0]), ...] = keypoints
                padded_gt_keypoints.append(new_keypoints)

            # pad zeros to masks
            if has_mask:
                # [n, img_h, img_w] -> [n, max_img_h, max_img_w]
                masks = torch.from_numpy(ground_truth_masks[b_ix])
                masks = F.pad(Variable(masks), pad_size, 'constant', 0).data.cpu()
                # [n, max_img_h, max_img_w] -> [max_num_gt_bboxes, max_img_h, max_img_w]
                if masks.shape[0] < max_num_gt_bboxes:
                    pad_masks = masks.new(max_num_gt_bboxes - masks.shape[0], max_img_h, max_img_w).zero_()
                    masks = torch.cat([masks, pad_masks], dim=0)
                padded_gt_masks.append(masks.numpy())

        padded_images = torch.cat(padded_images, dim = 0)
        unpad_image_sizes = np.stack(unpad_image_sizes, axis = 0)
        stack_fn = lambda x : np.stack(x, axis=0) if x else np.array([])
        padded_gt_bboxes = stack_fn(padded_gt_bboxes)
        padded_ig_bboxes = stack_fn(padded_ig_bboxes)
        padded_gt_keypoints = stack_fn(padded_gt_keypoints)
        padded_gt_masks = stack_fn(padded_gt_masks)

        #logger.debug('image.shape:{}'.format(padded_images.shape))
        #logger.debug('gt_box.shape:{}'.format(padded_gt_bboxes.shape))
        #logger.debug('image_info.shape:{}'.format(unpad_image_sizes.shape))
        #logger.debug('gt_kpts.shape:{}'.format(padded_gt_keypoints.shape))
        #logger.debug('gt_mask.shape:{}'.format(padded_gt_masks.shape))
        return [padded_images,
                unpad_image_sizes,
                padded_gt_bboxes,
                padded_ig_bboxes,
                padded_gt_keypoints,
                padded_gt_masks,
                filenames]


def validate(anno_file):
    from pycocotools.coco import COCO
    coco = COCO(anno_file)
    image_a = set()
    image_b = set()
    for anno in coco.anns.values():
        image_a.add(anno['image_id'])
        if anno['num_keypoints'] > 0:
            image_b.add(anno['image_id'])
    print('total images of person :{}\n'.format(len(image_a)))
    print('images with annotated keypoints:{}\n'.format(len(image_b)))

