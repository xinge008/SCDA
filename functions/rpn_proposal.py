#encoding: utf-8
from utils import bbox_helper
from utils import anchor_helper
from extensions import nms
import torch
import torch.nn.functional as F
import numpy as np
import logging
logger = logging.getLogger('global')

def to_np_array(x):
    if x is None:
        return None
    # if isinstance(x, Variable): x = x.data
    return x.cpu().data.numpy() if torch.is_tensor(x) else x

def compute_rpn_proposals(conv_cls, conv_loc, cfg, image_info):
    '''
    :argument
        cfg: configs
        conv_cls: FloatTensor, [batch, num_anchors * x, h, w], conv output of classification
        conv_loc: FloatTensor, [batch, num_anchors * 4, h, w], conv output of localization
        image_info: FloatTensor, [batch, 3], image size
    :returns
        proposals: Variable, [N, 5], 2-dim: batch_ix, x1, y1, x2, y2
    '''

    batch_size, num_anchors_4, featmap_h, featmap_w = conv_loc.shape
    # [K*A, 4]
    anchors_overplane = anchor_helper.get_anchors_over_plane(featmap_h, featmap_w,
                                                             cfg['anchor_ratios'], cfg['anchor_scales'], cfg['anchor_stride'])
    B = batch_size
    A = num_anchors = num_anchors_4 // 4
    assert(A * 4 == num_anchors_4)
    K = featmap_h * featmap_w

    cls_view = conv_cls.permute(0, 2, 3, 1).contiguous().view(B, K*A, -1).cpu().numpy()
    loc_view = conv_loc.permute(0, 2, 3, 1).contiguous().view(B, K*A, 4).cpu().numpy()
    if torch.is_tensor(image_info):
        image_info = image_info.cpu().numpy()

    #all_proposals = [bbox_helper.compute_loc_bboxes(anchors_overplane, loc_view[ix]) for ix in range(B)]
    # [B, K*A, 4]
    #pred_loc = np.stack(all_proposals, axis = 0)
    #pred_cls = cls_view
    batch_proposals = []
    pre_nms_top_n = cfg['pre_nms_top_n']
    for b_ix in range(B):
        scores = cls_view[b_ix, :, -1] # to compatible with sigmoid
        if pre_nms_top_n <= 0 or pre_nms_top_n > scores.shape[0]:
            order = scores.argsort()[::-1]
        else:
            inds = np.argpartition(-scores, pre_nms_top_n)[:pre_nms_top_n]
            order = np.argsort(-scores[inds])
            order = inds[order]
        loc_delta = loc_view[b_ix, order, :]
        loc_anchors = anchors_overplane[order, :]
        scores = scores[order]
        boxes = bbox_helper.compute_loc_bboxes(loc_anchors, loc_delta)
        boxes = bbox_helper.clip_bbox(boxes, image_info[b_ix])
        proposals = np.hstack([boxes, scores[:, np.newaxis]])
        proposals = proposals[(proposals[:, 2] - proposals[:, 0] + 1 >= cfg['roi_min_size'])
                            & (proposals[:, 3] - proposals[:, 1] + 1 >= cfg['roi_min_size'])]
        keep_index = nms(torch.from_numpy(proposals).float().cuda(), cfg['nms_iou_thresh']).numpy()
        if cfg['post_nms_top_n'] > 0:
            keep_index = keep_index[:cfg['post_nms_top_n']]
        proposals = proposals[keep_index]
        batch_ix = np.full(keep_index.shape, b_ix)
        proposals = np.hstack([batch_ix[:, np.newaxis], proposals])
        batch_proposals.append(proposals)
    batch_proposals = (torch.from_numpy(np.vstack(batch_proposals))).float()
    if batch_proposals.dim() < 2:
        batch_proposals.unsqueeze(dim=0)
    return batch_proposals
