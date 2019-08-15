#encoding:utf8
from utils import bbox_helper
from extensions import nms
import torch
import logging
import numpy as np
def to_np_array(x):
    if x is None:
        return None
    # if isinstance(x, Variable): x = x.data
    return x.cpu().data.numpy() if torch.is_tensor(x) else x

def compute_predicted_bboxes(rois, pred_cls, pred_loc, image_info, cfg):
    '''
    :param cfg: config
    :param rois: [N, k] k>=5, batch_ix, x1, y1, x2, y2
    :param pred_cls:[N, num_classes, 1, 1]
    :param pred_loc:[N, num_classes * 4, 1, 1]
    :param image_info:[N, 3]
    :return: bboxes: [M, 7], batch_ix, x1, y1, x2, y2, score, cls
    '''
    # logger = logging.getLogger('global')
    rois, pred_cls, pred_loc = map(to_np_array, [rois, pred_cls, pred_loc])
    N, num_classes = pred_cls.shape[0:2]
    B = max(rois[:, 0].astype(np.int32))+1
    assert(N == rois.shape[0])
    nmsed_bboxes = []
    for cls in range(1, num_classes):
        scores = pred_cls[:, cls].squeeze()
        deltas = pred_loc[:, cls*4:cls*4+4].squeeze()
        if cfg['bbox_normalize_stats_precomputed']:
            deltas = deltas * np.array(cfg['bbox_normalize_stds'])[np.newaxis, :]\
                     + np.array(cfg['bbox_normalize_means'])[np.newaxis, :]
        bboxes = bbox_helper.compute_loc_bboxes(rois[:,1:1+4], deltas)
        bboxes = np.hstack([bboxes, scores[:, np.newaxis]])
        # for each image, do nms
        for b_ix in range(B):
            rois_ix = np.where(rois[:, 0] == b_ix)[0]
            pre_scores = scores[rois_ix]
            pre_bboxes = bboxes[rois_ix]
            pre_bboxes[:, :4] = bbox_helper.clip_bbox(pre_bboxes[:,:4], image_info[b_ix])
            if cfg['score_thresh'] > 0:
                keep_ix = np.where(pre_scores > cfg['score_thresh'])[0]
                pre_scores = pre_scores[keep_ix]
                pre_bboxes = pre_bboxes[keep_ix]
            if pre_scores.size == 0: continue
            order = pre_scores.argsort()[::-1]
            pre_bboxes = pre_bboxes[order, :]
            keep_index = nms(torch.from_numpy(pre_bboxes).float().cuda(), cfg['nms_iou_thresh']).numpy()
            post_bboxes = pre_bboxes[keep_index]
            batch_ix = np.full(post_bboxes.shape[0], b_ix)
            batch_cls = np.full(post_bboxes.shape[0], cls)
            post_bboxes = np.hstack([batch_ix[:, np.newaxis], post_bboxes, batch_cls[:, np.newaxis]])
            nmsed_bboxes.append(post_bboxes)
    nmsed_bboxes = np.vstack(nmsed_bboxes)
    if cfg['top_n'] > 0:
        top_n_bboxes = []
        for b_ix in range(B):
            bboxes = nmsed_bboxes[nmsed_bboxes[:, 0] == b_ix]
            scores = bboxes[:, -2]
            order = scores.argsort()[::-1][:cfg['top_n']]
            bboxes = bboxes[order]
            top_n_bboxes.append(bboxes)
        nmsed_bboxes = np.vstack(top_n_bboxes)
    nmsed_bboxes = (torch.from_numpy(nmsed_bboxes)).float().cuda()
    return nmsed_bboxes
