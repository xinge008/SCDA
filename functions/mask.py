#encoding: utf-8
from utils import bbox_helper
# from utils.debug_helper import debugger
# import utils.visualize_helper as vis_helper

import torch
import numpy as np
import cv2
from PIL import Image
import logging
logger = logging.getLogger('global')


def to_np_array(x):
    if x is None:
        return None
    # if isinstance(x, Variable): x = x.data
    return x.cpu().data.numpy() if torch.is_tensor(x) else np.array(x)


def predict_masks(rois, heatmap, image_info, cfg = None):
    '''
    Args:
        rois: [R,k], k>=7, Variable or Tensor with shape of [R, k] (batch_index,x1,y1,x2,y2,socre,class,...)
            all rois must be inside images
        heatmap: [R, num_classes, label_h, label_w] Variable or tensor
        image_info: [batch_size, k2], k2 >= 3, (resized_image_h, resized_image_w, resize_scale)
        cfg: config
    Return:
        masks: list of np.array, mask results on resized_image
    '''
    rois = to_np_array(rois)
    heatmap = to_np_array(heatmap)
    assert(rois.shape[0] == heatmap.shape[0])
    R = rois.shape[0]
    masks = []
    for r_ix in range(R):
        roi = rois[r_ix]
        b_ix, x1, y1, x2, y2, score, cls = map(int, roi[:7])
        del score
        roi_w = x2 - x1 + 1
        roi_h = y2 - y1 + 1
        image_h, image_w = map(int, image_info[b_ix][:2])
        # mask = cv2.resize(heatmap[r_ix, cls], (roi_w, roi_h))
        mask = np.array(Image.fromarray(heatmap[r_ix, cls]).resize((roi_w, roi_h)))
        image = np.zeros((image_h, image_w), dtype = np.float32)
        image[y1:y1+roi_h, x1:x1+roi_w] = mask
        masks.append(image)
    return masks

def generate_mask_labels(rois, masks, mask_h, mask_w):
    '''
    Args:
        rois: [N, k], k>=4 (x1,y1,x2,y2, ...)
        masks: [N, padded_image_h, padded_image_w], binary map
        mask_h, mask_w: output size
    return:
        mask_labels: [N, mask_h, mask_w], binary map
    '''
    rois = rois.astype(np.int32)
    assert(rois.shape[0] == masks.shape[0])
    N = rois.shape[0]
    mask_labels = []
    for r_ix, roi in enumerate(rois):
        x1, y1, x2, y2 = roi
        assert(x1 < x2 and y1 < y2)
        mask = masks[r_ix]
        mask = cv2.resize(mask[y1:y2, x1:x2], (mask_w, mask_h))
        mask_labels.append(mask)
    mask_labels = np.stack(mask_labels, axis=0).astype(np.int32)
    return mask_labels

def compute_mask_targets(proposals,
                         cfg,
                         ground_truth_bboxes,
                         ground_truth_masks,
                         image_info,
                         ignore_regions = None):
    '''
    Args:
        proposals:[N, k], k>=5(b_ix, x1,y1,x2,y2, ...)
        ground_truth_bboxes: [batch_size, max_gts, k], k>=5(x1,y1,x2,y2,label)
        ground_truth_masks: [batch_size, max_gts, image_h, image_w]
        image_info: [batch_size, 3], (resized_image_h, resized_image_w, resize_scale)
    Return:
        batch_rois: [R, 5] (b_ix, x1,y1,x2,y2)
        batch_kpt_labels: [R, num_classes, label_h, label_w]
    '''
    proposals_device = proposals.device
    proposals = to_np_array(proposals)
    ground_truth_bboxes = to_np_array(ground_truth_bboxes)
    ground_truth_masks = to_np_array(ground_truth_masks)
    image_info = to_np_array(image_info)
    ignore_regions = to_np_array(ignore_regions)


    B = ground_truth_bboxes.shape[0]
    batch_rois = []
    batch_mask_labels = []

    for b_ix in range(B):
        rois = proposals[proposals[:, 0] == b_ix][:, 1:1+4]
        gts = ground_truth_bboxes[b_ix]
        masks = ground_truth_masks[b_ix]
        # kick out padded gts
        keep_ix = np.where(gts[:, 2] > gts[:, 1] + 1)[0]
        if keep_ix.size == 0: continue
        gts = gts[keep_ix]
        masks = masks[keep_ix]
        if cfg['append_gts']:
            rois = np.vstack([rois, gts[:, :4]])
        rois = bbox_helper.clip_bbox(rois.astype(np.int32), image_info[b_ix].astype(np.int32))
        R = rois.shape[0]
        G = gts.shape[0]
        if R == 0 or G == 0: continue
        # [R, G]
        overlaps = bbox_helper.bbox_iou_overlaps(rois, gts)
        # [R]
        # (i): a roi that has an IoU higher than than positive_iou_thresh is postive
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        pos_r_ix = np.where(max_overlaps > cfg['positive_iou_thresh'])[0]
        pos_g_ix = argmax_overlaps[pos_r_ix]

        # sampling
        num_positives = pos_r_ix.shape[0]
        if num_positives == 0: continue
        if cfg['batch_size_per_image'] > 0 and num_positives > cfg['batch_size_per_image']:
            keep_ix = np.random.choice(num_positives, size = cfg['batch_size_per_image'], replace = False)
            pos_r_ix = pos_r_ix[keep_ix]
            pos_g_ix = pos_g_ix[keep_ix]

        # gather positive bboxes and related masks
        pos_rois = rois[pos_r_ix]
        pos_target_classes = gts[pos_g_ix][:, 4].astype(np.int64)
        pos_target_masks = masks[pos_g_ix]
        N = pos_rois.shape[0]
        pos_mask_labels = generate_mask_labels(pos_rois,
                pos_target_masks, cfg['label_h'], cfg['label_w'])

        mask_labels = -np.ones((N, cfg['num_classes'], cfg['label_h'], cfg['label_w']))
        mask_labels[range(N), pos_target_classes, ...] = pos_mask_labels

        batch_idx = np.full((N, 1), b_ix)
        pos_rois = np.hstack([batch_idx, pos_rois, pos_target_classes[:, np.newaxis]])

        batch_rois.append(pos_rois)
        batch_mask_labels.append(mask_labels)
    if len(batch_rois) == 0:
        # if there's no positive rois, pad zeros
        n = 1
        batch_rois = np.zeros((n,5), dtype = np.float32)
        batch_mask_labels = -np.ones((n, cfg['num_classes'], cfg['label_h'], cfg['label_w']), dtype = np.float32)
    else:
        batch_rois = np.vstack(batch_rois)
        batch_mask_labels = np.vstack(batch_mask_labels)

    # debug
    #import os
    #import torch.distributed as dist
    #vis_mask = 'vis_mask'
    #if not os.path.exists(vis_mask):
    #    os.makedirs(vis_mask)
    #for i, roi in enumerate(batch_rois):
    #    b_ix, x1, y1, x2, y2, cls = map(int, roi[:6])
    #    roi_w = x2 - x1
    #    roi_h = y2 - y1
    #    img = debugger.get_image(b_ix).copy()
    #    filename = debugger.get_filename(b_ix).split('/')[-1].split('.')[0]
    #    mask = batch_mask_labels[i, cls]
    #    mask = cv2.resize(mask, (roi_w, roi_h)) * 100
    #    img[y1:y2, x1:x2, ...] += mask[..., np.newaxis]
    #    vis_helper.draw_bbox(img, roi[1:1+4])
    #    cv2.imwrite('vis_mask/{0}_{1}.jpg'.format(filename, i), img)
    cuda_device = proposals_device
    f = lambda x: (torch.from_numpy(x)).to(cuda_device)
    batch_rois = f(batch_rois).float()
    batch_mask_labels = f(batch_mask_labels).float()
    return batch_rois, batch_mask_labels

from sklearn.cluster import KMeans

def proposals_to_centers(proposals):
    """
    :param proposals: [N, 5], (b_ix, x1, y1, x2, y2)
    :return: centers [N, 2], (b_ix, center_x, center_y)
    """
    cx = (proposals[:, 3] + proposals[:, 1]) / 2.0
    cy = (proposals[:, 4] + proposals[:, 2]) / 2.0
    center = np.vstack([cx, cy]).transpose()
    return center

def compute_cluster_targets(proposals, features, N_cluster=4, threshold=128):
    '''
    Args:
        proposals:[N, k], k>=5(b_ix, x1,y1,x2,y2, ...), N = 512
        features: [N, 4096],
    Return:
        batch_rois: [N_cluster, 128, 4096]
        batch_cluster_center: [N_cluster, 2], (center_x, center_y)
    '''
    proposals_np = to_np_array(proposals)
    features_np = to_np_array(features)
    centers = proposals_to_centers(proposals_np)

    """
    KMeans part
    """
    kmeans = KMeans(n_clusters=N_cluster, random_state=0).fit(centers)

    cluster_center = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    batch_rois_cluster = []
    for cluster_idx in range(0, N_cluster):
        keep_ix = np.where(cluster_labels[:] == cluster_idx)[0]

        if keep_ix.shape[0] < threshold:
            keep_ix_new = np.random.choice(keep_ix.shape[0], threshold, replace=True)
            keep_ix2 = keep_ix[keep_ix_new]
            batch_rois_tmp = features_np[keep_ix2]
        else:
            keep_ix2 = keep_ix[0:threshold]
            batch_rois_tmp = features_np[keep_ix2]


        # batch_rois_tmp = features[keep_ix]
        batch_rois_cluster.append(batch_rois_tmp)

    batch_rois_cluster = np.stack(batch_rois_cluster, axis=0) # (N_cluster, threshold, 4096)



    f = lambda x: (torch.from_numpy(x)).float().cuda().contiguous()
    batch_rois_cluster = f(batch_rois_cluster)
    # batch_mask_labels = f(batch_mask_labels).float()
    return batch_rois_cluster, cluster_center

