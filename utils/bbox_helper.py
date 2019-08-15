#encoding: utf-8

import numpy as np
import warnings
from extensions._cython_bbox import cython_bbox


def bbox_iou_overlaps(b1, b2):
    return cython_bbox.bbox_overlaps(b1.astype(np.float32), b2.astype(np.float32))
    '''
    :argument
        b1,b2: [n, k], k>=4, x1,y1,x2,y2,...
    :returns
        intersection-over-union pair-wise.
    '''
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    union_area1 = area1.reshape(-1, 1) + area2.reshape(1, -1)
    union_area2 = (union_area1 - inter_area)
    return inter_area / np.maximum(union_area2, 1)
    
def bbox_iof_overlaps(b1, b2):
    '''
    :argument
        b1,b2: [n, k], k>=4 with x1,y1,x2,y2,....
    :returns
        intersection-over-former-box pair-wise
    '''
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    # area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    inter_xmin = np.maximum(b1[:, 0].reshape(-1, 1), b2[:, 0].reshape(1, -1))
    inter_ymin = np.maximum(b1[:, 1].reshape(-1, 1), b2[:, 1].reshape(1, -1))
    inter_xmax = np.minimum(b1[:, 2].reshape(-1, 1), b2[:, 2].reshape(1, -1))
    inter_ymax = np.minimum(b1[:, 3].reshape(-1, 1), b2[:, 3].reshape(1, -1))
    inter_h = np.maximum(inter_xmax - inter_xmin, 0)
    inter_w = np.maximum(inter_ymax - inter_ymin, 0)
    inter_area = inter_h * inter_w
    return inter_area / np.maximum(area1[:,np.newaxis], 1)

def center_to_corner(boxes):
    '''
    :argument
        boxes: [N, 4] of center_x, center_y, w, h
    :returns
        boxes: [N, 4] of xmin, ymin, xmax, ymax
    '''
    xmin = boxes[:, 0] - boxes[:, 2] / 2.
    ymin = boxes[:, 1] - boxes[:, 3] / 2.
    xmax = boxes[:, 0] + boxes[:, 2] / 2.
    ymax = boxes[:, 1] + boxes[:, 3] / 2.
    return np.vstack([xmin, ymin, xmax, ymax]).transpose()

def corner_to_center(boxes):
    '''
        inverse of center_to_corner
    '''
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.
    w = (boxes[:, 2] - boxes[:, 0])
    h = (boxes[:, 3] - boxes[:, 1])
    return np.vstack([cx, cy, w, h]).transpose()

def compute_loc_targets(raw_bboxes, gt_bboxes):
    '''
    :argument
        raw_bboxes, gt_bboxes:[N, k] first dim must be equal
    :returns
        loc_targets:[N, 4]
    '''
    bb = corner_to_center(raw_bboxes) # cx, cy, w, h
    gt = corner_to_center(gt_bboxes)
    assert (np.all(bb[:, 2] > 0))
    assert (np.all(bb[:, 3] > 0))
    trgt_ctr_x = (gt[:, 0] - bb[:, 0]) / bb[:, 2]
    trgt_ctr_y = (gt[:, 1] - bb[:, 1]) / bb[:, 3]
    trgt_w = np.log(gt[:, 2] / bb[:, 2])
    trgt_h = np.log(gt[:, 3] / bb[:, 3])
    return np.vstack([trgt_ctr_x, trgt_ctr_y, trgt_w, trgt_h]).transpose()


def compute_loc_bboxes(raw_bboxes, deltas):
    '''
    :argument
        raw_bboxes, delta:[N, k] first dim must be equal
    :returns
        bboxes:[N, 4]
    '''
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bb = corner_to_center(raw_bboxes) # cx, cy, w, h
        dt_cx = deltas[:, 0] * bb[:, 2] + bb[:, 0]
        dt_cy = deltas[:, 1] * bb[:, 3] + bb[:, 1]
        dt_w = np.exp(deltas[:, 2]) * bb[:, 2]
        dt_h = np.exp(deltas[:, 3]) * bb[:, 3]
        dt = np.vstack([dt_cx, dt_cy, dt_w, dt_h]).transpose()
        return center_to_corner(dt)

def clip_bbox(bbox, img_size):
    h, w = img_size[:2]
    bbox[:, 0] = np.clip(bbox[:, 0], 0, w - 1)
    bbox[:, 1] = np.clip(bbox[:, 1], 0, h - 1)
    bbox[:, 2] = np.clip(bbox[:, 2], 0, w - 1)
    bbox[:, 3] = np.clip(bbox[:, 3], 0, h - 1)
    return bbox

def compute_recall(box_pred, box_gt):
    n_gt = box_gt.shape[0]
    if box_pred.size == 0 or n_gt == 0:
        return 0, n_gt
    ov = bbox_iou_overlaps(box_gt, box_pred)
    max_ov = np.max(ov, axis=1)
    idx = np.where(max_ov>0.5)[0]
    n_rc = idx.size
    return n_rc, n_gt

