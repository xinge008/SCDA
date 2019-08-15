import torch
import numpy as np
import logging
#from utils.timer import Timer

def to_np_array(x):
    if x is None:
        return None
    # if isinstance(x, Variable): x = x.data
    return x.cpu().data.numpy() if torch.is_tensor(x) else np.array(x)

def get_rois_target_levels(levels, base_scale, base_level, rois):
    '''assign proposals to different level feature map to roi pooling
        Args:
            rois: [R, 5], batch_ix,x1,y1,x2,y2
            levels: [L], levels. e.g.[2,3,4,5,6]
    '''
    rois = to_np_array(rois)
    w = rois[:, 3] - rois[:, 1] + 1
    h = rois[:, 4] - rois[:, 2] + 1
    scale = (w * h)**0.5
    eps = 1e-6
    target_levels = np.floor(base_level + np.log2(scale/base_scale+eps)).astype(np.int32)
    min_level, max_level = min(levels), max(levels)
    return np.clip(target_levels, min_level, max_level)

def get_rois_by_level(levels, base_scale, base_level, rois):
    rois = to_np_array(rois)
    target_lvls = get_rois_target_levels(levels, base_scale, base_level, rois)
    rois_by_level, rois_ix_by_level = [], []
    for lvl in levels:
        ix = np.where(target_lvls == lvl)[0]
        rois_by_level.append(rois[ix])
        rois_ix_by_level.append(ix)
    return rois_by_level, rois_ix_by_level

def assign_args_by_level(levels, base_scale, base_level, rois, *args):
    '''
        Args:
            rois: [R, 5], batch_ix,x1,y1,x2,y2
            levels: [L], levels. e.g.[2,3,4,5,6]
        return:
            args by level
    '''
    args_by_level = []
    rois = to_np_array(rois)
    rois_by_level, rois_ix_by_level = \
            get_rois_by_level(levels, base_scale, base_level, rois)

    args_by_level.append(rois_by_level)
    for arg in args:
        # assign arg to each level
        arg = to_np_array(arg)
        arg_by_level = []
        for ix in rois_ix_by_level:
            arg_by_level.append(arg[ix])
        args_by_level.append(arg_by_level)
    return args_by_level

def get_proposals_assign(proposals, base_scale=224, layer_index=4):
    '''
    :arguement
        proposals:[N, k], k>=5, batch_idx, x1, y1, x2, y2
        base_scale: base scale
        layer_index: the layer RoI with wxh=224x22 should be mapped into
    returns:
        p*: [N, 5]
    '''
    #logger = logging.getLogger('global')
    #p = map(lambda x: x.cpu().numpy() if torch.is_tensor(x) else x, [proposals])
    p = to_np_array(proposals)
    w = p[:,3] - p[:,1] + 1
    h = p[:,4] - p[:,2] + 1
    area = (w*h)**0.5
    k = np.floor(layer_index + np.log2(area/base_scale))
    p2 = p[k <= 2] 
    p3 = p[k == 3] 
    p4 = p[k == 4] 
    p5 = p[k >= 5] 
    return p2, p3, p4, p5

def get_rois_assign(rois, cls_targets, loc_targets, loc_weights, base_scale=224, layer_index=4):
    #logger = logging.getLogger('global')
    #T = Timer()
    #roi = rois.data.cpu().numpy()
    #cls_t = cls_targets.data.cpu().numpy()
    #loc_t = loc_targets.data.cpu().numpy()
    #loc_w = loc_weights.data.cpu().numpy()
    roi = rois
    cls_t = cls_targets
    loc_t = loc_targets
    loc_w = loc_weights

    w = roi[:,3] - roi[:,1] + 1
    h = roi[:,4] - roi[:,2] + 1
    area = (w*h)**0.5
    k = np.floor(layer_index + np.log2(area/base_scale))
    p2 = k <= 2
    p3 = k == 3
    p4 = k == 4
    p5 = k >= 5
    roi_new = []
    cls_t_new = []
    loc_t_new = []
    loc_w_new = []
    for p in [p2, p3, p4, p5]:
         roi_new.append(roi[p])
         if np.where(p==True)[0].size > 0:
             cls_t_new.append(cls_t[p])
             loc_t_new.append(loc_t[p])
             loc_w_new.append(loc_w[p])

    cuda_device = rois.device
    f = lambda x: (torch.from_numpy(x)).cuda()
    cls_ts = f(np.concatenate(cls_t_new)).long()
    loc_ts = f(np.vstack(loc_t_new)).float()
    loc_ws = f(np.vstack(loc_w_new)).float()
    return roi_new, cls_ts, loc_ts, loc_ws
