#encoding: utf-8
from utils import bbox_helper
# from utils.debug_helper import debugger
import numpy as np
import torch
import logging

logger = logging.getLogger('global')
history = [0, 0]

def to_np_array(x):
    if x is None:
        return None
    # if isinstance(x, Variable): x = x.data
    return x.cpu().data.numpy() if torch.is_tensor(x) else x

def compute_proposal_targets(proposals, cfg, ground_truth_bboxes, image_info, ignore_regions = None, use_ohem = False):
    '''
    :argument
        proposals:[N, k], k>=5, batch_idx, x1, y1, x2, y2
        ground_truth_bboxes: [batch, max_num_gts, k], k>=5, x1,y1,x2,y2,label
    returns:
        rois: [N, 5]:
        cls_targets: [N, num_classes]
        loc_targets, loc_weights: [N, num_classes * 4]
    '''
    proposals, ground_truth_bboxes, image_info, ignore_regions = \
        map(to_np_array, [proposals, ground_truth_bboxes, image_info, ignore_regions])
    B = ground_truth_bboxes.shape[0]
    logger.debug('proposals.shape:{}'.format(proposals.shape))
    logger.debug('ground_truth_bboxes.shape:{}'.format(ground_truth_bboxes.shape))
    batch_rois = []
    batch_labels = []
    batch_loc_targets = []
    batch_loc_weights = []
    for b_ix in range(B):
        rois = proposals[proposals[:, 0] == b_ix][:, 1:1+4]
        gts = ground_truth_bboxes[b_ix]
        # kick out padded empty ground truth bboxes
        #gts = gts[gts[:, 2] > gts[:, 0] + 1]
        gts = gts[(gts[:,2] > gts[:,0]+1) & (gts[:,3] > gts[:,1]+1)]
        if cfg['append_gts']:
            rois = np.vstack([rois, gts[:, :4]])
        rois = bbox_helper.clip_bbox(rois, image_info[b_ix])
        R = rois.shape[0]
        G = gts.shape[0]
        if R == 0 or G == 0: continue
        #[R, G]
        overlaps = bbox_helper.bbox_iou_overlaps(rois, gts)

        # (i) the anchor with the highest Intersection-over-Union (IoU)
        # overlap with a ground-truth box is positive
        # [G]
        #gt_max_overlaps = overlaps.max(axis=0)
        #gt_max_overlaps[gt_max_overlaps < 0.1] = -1
        #gt_pos_r_ix, gt_pos_g_ix = np.where(overlaps == gt_max_overlaps[np.newaxis, :])

        # (ii) an anchor that has an IoU overlap higher than positive_iou_thresh
        # with any ground-truth box is positive
        # [R]
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        pos_r_ix = np.where(max_overlaps > cfg['positive_iou_thresh'])[0]
        pos_g_ix = argmax_overlaps[pos_r_ix]

        # merge pos_r_ix & gt_pos_b_ix
        #pos_r_ix = np.concatenate([pos_r_ix, gt_pos_r_ix])
        #pos_g_ix = np.concatenate([pos_g_ix, gt_pos_g_ix])
        # remove duplicate positives
        pos_r_ix, return_index = np.unique(pos_r_ix, return_index=True)
        pos_g_ix = pos_g_ix[return_index]

        # (iii) We assign a negative label to a non-positive anchor if its IoU ratio
        # is between [negative_iou_thresh_lo, negative_iou_thresh_low] for all ground-truth boxes
        neg_r_ix = np.where((max_overlaps < cfg['negative_iou_thresh_hi'])
                            & (max_overlaps >= cfg['negative_iou_thresh_lo']))[0]

        # remove negatives which located in ignore regions
        if ignore_regions is not None:
            cur_ignore = ignore_regions[b_ix]
            # remove padded ignore regions
            cur_ignore = cur_ignore[cur_ignore[:, 2] - cur_ignore[:, 0] > 1]
            if cur_ignore.shape[0] > 0:
                iof_overlaps = bbox_helper.bbox_iof_overlaps(rois, cur_ignore)
                max_iof_overlaps = iof_overlaps.max(axis=1)  # [B, K*A]
                ignore_rois_ix = np.where(max_iof_overlaps > cfg['ignore_iou_thresh'])[0]
                neg_r_ix = np.array(list(set(neg_r_ix) - set(ignore_rois_ix)))

        # remove positives(rule (i)) from negatives
        neg_r_ix = np.array(list(set(neg_r_ix) - set(pos_r_ix)))

        #sampling
        num_positives = len(pos_r_ix)
        
        batch_size_per_image = cfg['batch_size']

        # keep all pos and negs if use OHEM
        if not use_ohem:
            num_pos_sampling = int(cfg['positive_percent'] * batch_size_per_image)
            if num_pos_sampling < num_positives:
                keep_ix = np.random.choice(num_positives, size = num_pos_sampling, replace = False)
                pos_r_ix = pos_r_ix[keep_ix]
                pos_g_ix = pos_g_ix[keep_ix]
                num_positives = num_pos_sampling

            num_negatives = len(neg_r_ix)
            num_neg_sampling = batch_size_per_image - num_positives
            if num_neg_sampling < num_negatives:
                keep_ix = np.random.choice(num_negatives, size = num_neg_sampling, replace = False)
                neg_r_ix = neg_r_ix[keep_ix]
                num_negatives = num_neg_sampling
            #else:
            #    keep_ix = np.random.choice(num_negatives, size = num_neg_sampling, replace = True)
            #    neg_r_ix = neg_r_ix[keep_ix]
            #    num_negatives = num_neg_sampling

        # convert neg_r_ix, pos_r_ix and pos_g_ix from np.array to list in case of *_ix == np.array([])
        # which can't index np.array
        pos_r_ix = list(pos_r_ix)
        pos_g_ix = list(pos_g_ix)
        neg_r_ix = list(neg_r_ix)
        # gather positives, matched gts, and negatives
        pos_rois = rois[pos_r_ix]
        pos_target_gts = gts[pos_g_ix]
        neg_rois = rois[neg_r_ix]
        rois_sampling = np.vstack([pos_rois, neg_rois])
        num_pos, num_neg = pos_rois.shape[0], neg_rois.shape[0]
        num_sampling = num_pos + num_neg
        
        # generate targets
        pos_labels = pos_target_gts[:,4].astype(np.int32)
        neg_labels = np.zeros(num_neg)
        labels = np.concatenate([pos_labels, neg_labels]).astype(np.int32)
        
        loc_targets = np.zeros([num_sampling, cfg['num_classes'], 4])
        loc_weights = np.zeros([num_sampling, cfg['num_classes'], 4])
        pos_loc_targets = bbox_helper.compute_loc_targets(pos_rois, pos_target_gts)
        if cfg['bbox_normalize_stats_precomputed']:
            pos_loc_targets = (pos_loc_targets - np.array(cfg['bbox_normalize_means'])[np.newaxis, :]) \
                              / np.array(cfg['bbox_normalize_stds'])[np.newaxis, :]
        loc_targets[range(num_pos), pos_labels, :] = pos_loc_targets
        loc_weights[range(num_pos), pos_labels, :] = 1
        loc_targets = loc_targets.reshape([num_sampling, -1])
        loc_weights = loc_weights.reshape([num_sampling, -1])

        batch_ix = np.full(rois_sampling.shape[0], b_ix)
        rois_sampling = np.hstack([batch_ix[:, np.newaxis], rois_sampling])

        if rois_sampling.shape[0] < batch_size_per_image:
            rep_num = batch_size_per_image - rois_sampling.shape[0]
            rep_index = np.random.choice(rois_sampling.shape[0], size=rep_num, replace=True)
            rois_sampling = np.vstack([rois_sampling, rois_sampling[rep_index]])
            labels = np.concatenate([labels, labels[rep_index]])
            loc_targets = np.vstack([loc_targets, loc_targets[rep_index]])
            loc_weights = np.vstack([loc_weights, loc_weights[rep_index]])

        batch_rois.append(rois_sampling)
        batch_labels.append(labels)
        batch_loc_targets.append(loc_targets)
        batch_loc_weights.append(loc_weights)

    pos_num = np.where(np.concatenate(batch_labels) > 0)[0].shape[0]
    neg_num = np.concatenate(batch_labels).shape[0] - pos_num
    history[0] += pos_num
    history[1] += neg_num
    history_pos, history_neg = history
    pos_percent = history_pos / (history_neg + history_pos)
    neg_percent = history_neg / (history_neg + history_pos)
    logger.debug('proposal_target(pos/neg): %d=%d+%d, history ratio:%.5f/%.5f'
              % (pos_num + neg_num, pos_num, neg_num, pos_percent, neg_percent))

    batch_rois = (torch.from_numpy(np.vstack(batch_rois))).float().cuda().contiguous()
    batch_labels = (torch.from_numpy(np.concatenate(batch_labels))).long().cuda().contiguous()
    batch_loc_targets = (torch.from_numpy(np.vstack(batch_loc_targets))).float().cuda().contiguous()
    batch_loc_weights = (torch.from_numpy(np.vstack(batch_loc_weights))).float().cuda().contiguous()

    return batch_rois, batch_labels, batch_loc_targets, batch_loc_weights
