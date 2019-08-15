#encoding: utf-8
# from utils.debug_helper import debugger
from utils import bbox_helper
from utils import anchor_helper
import numpy as np
import torch
import logging
logger = logging.getLogger('global')

def to_np_array(x):
    if x is None:
        return None
    # if isinstance(x, Variable): x = x.data
    return x.cpu().data.numpy() if torch.is_tensor(x) else np.array(x)

def compute_anchor_targets(feature_size, cfg, ground_truth_bboxes, image_info, ignore_regions = None):
    r'''
    :argument
        cfg.keys(): {
            'anchor_ratios', anchor_scales, anchor_stride,
            negative_iou_thresh, ignore_iou_thresh,positive_iou_thresh,
            positive_percent, rpn_batch_size
        }
        feature_size: IntTensor, [4]. i.e. batch, num_anchors * 4, height, width
        ground_truth_bboxes: FloatTensor, [batch, max_num_gt_bboxes, 5]
        image_info: FloatTensor, [batch, 3]
        ignore_regions: FloatTensor, [batch, max_num_ignore_regions, 4]
    :returns
        cls_targets: Variable, [batch, num_anchors * 1, height, width]
        loc_targets, loc_masks: Variable, [batch, num_anchors * 4, height, width]
    '''
    cuda_device = ground_truth_bboxes.device
    ground_truth_bboxes, image_info, ignore_regions = \
        map(to_np_array, [ground_truth_bboxes, image_info, ignore_regions])

    batch_size, num_anchors_4, featmap_h, featmap_w = feature_size
    num_anchors = num_anchors_4 // 4
    assert(num_anchors * 4 == num_anchors_4)
    # [K*A, 4]
    anchors_overplane = anchor_helper.get_anchors_over_plane(
            featmap_h, featmap_w, cfg['anchor_ratios'], cfg['anchor_scales'],
            cfg['anchor_stride'])

    B = batch_size
    A = num_anchors
    K = featmap_h * featmap_w
    G = ground_truth_bboxes.shape[1]

    # compute overlaps between anchors and gt_bboxes within each batch
    # shape: [B, K*A, G]
    overlaps = np.stack([bbox_helper.bbox_iou_overlaps(anchors_overplane,
                                                       ground_truth_bboxes[ix]) for ix in range(B)], axis = 0)

    # shape of [B, K*A]
    argmax_overlaps = overlaps.argmax(axis = 2)
    max_overlaps = overlaps.max(axis = 2)

    # [B, G]
    gt_max_overlaps = overlaps.max(axis=1)
    # ignore thoese gt_max_overlap too small
    gt_max_overlaps[gt_max_overlaps < 0.1] = -1
    gt_argmax_b_ix, gt_argmax_ka_ix, gt_argmax_g_ix = \
        np.where(overlaps == gt_max_overlaps[:, np.newaxis, :])
    # match each anchor to the ground truth bbox
    argmax_overlaps[gt_argmax_b_ix, gt_argmax_ka_ix] = gt_argmax_g_ix
    
    labels = np.empty([B, K*A], dtype=np.int64)
    labels.fill(-1)
    labels[max_overlaps < cfg['negative_iou_thresh']] = 0

    # remove negatives located in ignore regions
    if ignore_regions is not None:
        iof_overlaps = np.stack([bbox_helper.bbox_iof_overlaps
                                     (anchors_overplane, ignore_regions[ix]) for ix in range(B)], axis=0)
        max_iof_overlaps = iof_overlaps.max(axis=2)  # [B, K*A]
        labels[max_iof_overlaps > cfg['ignore_iou_thresh']] = -1

    labels[gt_argmax_b_ix, gt_argmax_ka_ix] = 1
    labels[max_overlaps > cfg['positive_iou_thresh']] = 1
    
    # sampling
    num_pos_sampling = int(cfg['positive_percent'] * cfg['rpn_batch_size'] * batch_size)
    pos_b_ix, pos_ka_ix = np.where(labels > 0)
    num_positives = len(pos_b_ix)
    if num_positives > num_pos_sampling:
        remove_ix = np.random.choice(num_positives, size = num_positives - num_pos_sampling, replace = False)
        labels[pos_b_ix[remove_ix], pos_ka_ix[remove_ix]] = -1
        num_positives = num_pos_sampling
    num_neg_sampling = cfg['rpn_batch_size'] * batch_size - num_positives
    neg_b_ix, neg_ka_ix = np.where(labels == 0)
    num_negatives = len(neg_b_ix)
    if num_negatives > num_neg_sampling:
        remove_ix = np.random.choice(num_negatives, size = num_negatives - num_neg_sampling, replace = False)
        labels[neg_b_ix[remove_ix], neg_ka_ix[remove_ix]] = -1
   
    pos_b_ix, pos_ka_ix = np.where(labels > 0)
    pos_anchors = anchors_overplane[pos_ka_ix, :]

    pos_target_ix = argmax_overlaps[pos_b_ix, pos_ka_ix]
    pos_target_gt = ground_truth_bboxes[pos_b_ix, pos_target_ix]
    pos_loc_targets = bbox_helper.compute_loc_targets(pos_anchors, pos_target_gt)

    loc_targets = np.zeros([B, K*A, 4], dtype = np.float32)
    loc_targets[pos_b_ix, pos_ka_ix, :] = pos_loc_targets
    # loc_weights = np.zeros([B, K*A, 4])
    loc_masks = np.zeros([B, K*A, 4], dtype = np.float32)
    loc_masks[pos_b_ix, pos_ka_ix, :] = 1.

    # transpose to match the predicted convolution shape

    cls_targets = torch.from_numpy(labels).long().view(B, featmap_h, featmap_w, A).permute(0, 3, 1, 2).cuda().contiguous()
    loc_targets = torch.from_numpy(loc_targets).float().view(B, featmap_h, featmap_w, A * 4).permute(0, 3, 1, 2).cuda().contiguous()
    loc_masks = torch.from_numpy(loc_masks).float().view(B, featmap_h, featmap_w, A * 4).permute(0, 3, 1, 2).cuda().contiguous()
    loc_nomalizer = max(1,len(np.where(labels >= 0)[0]))
    logger.debug('positive anchors:%d' % len(pos_b_ix))
    return cls_targets, loc_targets, loc_masks, loc_nomalizer
