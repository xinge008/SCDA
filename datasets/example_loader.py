#encoding: utf-8

import torch
import torch.nn.functional as F
import numpy as np
import logging

class ExampleDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, pin_memory=False, drop_last=False):
        super(ExampleDataLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                                        num_workers, self._collate_fn, pin_memory, drop_last)
    def _collate_fn(self, batch):
        batch_size = len(batch)

        zip_batch = list(zip(*batch))
        images = zip_batch[0]
        unpad_image_sizes = zip_batch[1]
        ground_truth_bboxes = zip_batch[2]
        ignores = zip_batch[3]
        filenames = zip_batch[4]

        max_img_h = max([_.shape[-2] for _ in images])
        max_img_w = max([_.shape[-1] for _ in images])
        max_num_gt_bboxes = max([_.shape[0] for _ in ground_truth_bboxes])
        max_num_ig_bboxes = max([_.shape[0] for _ in ignores])


        padded_images = []
        padded_gt_bboxes = []
        padded_ig_bboxes = []
        for b_ix in range(batch_size):
            img = images[b_ix]
            # pad zeros to right bottom of each image
            pad_size = (0, max_img_w - img.shape[-1], 0, max_img_h - img.shape[-2])
            padded_images.append(F.pad(img, pad_size, 'constant', 0).data.cpu())

            # pad zeros to gt_bboxes
            gt_bboxes = ground_truth_bboxes[b_ix].numpy()
            new_gt_bboxes = np.zeros([max_num_gt_bboxes, gt_bboxes.shape[-1]])
            new_gt_bboxes[range(gt_bboxes.shape[0]), :] = gt_bboxes
            padded_gt_bboxes.append(new_gt_bboxes)

            # pad zeros to ig_bboxes
            ig_bboxes = ignores[b_ix].numpy()
            new_ig_bboxes = np.zeros([max_num_ig_bboxes, ig_bboxes.shape[-1]])
            new_ig_bboxes[range(ig_bboxes.shape[0]), :] = ig_bboxes
            padded_ig_bboxes.append(new_ig_bboxes)

        padded_images = images = torch.cat(padded_images, dim = 0)
        padded_gt_bboxes = torch.from_numpy(np.stack(padded_gt_bboxes, axis = 0))
        padded_ig_bboxes = torch.from_numpy(np.stack(padded_ig_bboxes, axis = 0))
        unpad_image_sizes = torch.stack(unpad_image_sizes, dim = 0)
        #logger = logging.getLogger('global')
        #logger.debug('{0},{1},{2}'.format(padded_images.shape, padded_gt_bboxes.shape, unpad_image_sizes.shape))
        return padded_images, unpad_image_sizes, padded_gt_bboxes, padded_ig_bboxes, filenames
