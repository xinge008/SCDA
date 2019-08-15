#encoding:utf8

from utils import bbox_helper
try:
    from graphviz import Digraph
except Exception as e:
    print(e)
import torch
import numpy as np
import cv2
import os

classes = [
    '__background__',  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

def draw_bbox(img, bbox, color = (255,0,0)):
    box = np.array(bbox).astype(np.int32)
    return cv2.rectangle(img, tuple(box[0:2]), tuple(box[2:4]), color)

def draw_keypoint(img, keypoints, color = (255,0,0)):
    kpts = keypoints.reshape(-1, 2).astype(np.int32)
    for k in range(kpts.shape[0]):
        if k&1:
            cv2.circle(img, tuple(kpts[k]), 2, color, thickness=2) # left parts:blue
        else:
            cv2.circle(img, tuple(kpts[k]), 2, color[::-1], thickness=2) # right parts: red
    return img
def draw_mask(img, mask, thresh = 0.5):
    assert img.shape == mask.shape, 'img.shape:{} vs mask.shape'.format(img.shape, mask.shape)
    mask = (mask > thresh).astype(np.uint8) * 250
    img *= 0.5
    img += mask[..., np.newaxis] * 0.5
    return img


def vis_results(results_dir,image_info, bboxes, keypoints, masks, heatmap, class_names):
    from utils.debug_helper import debugger
    import logging
    logger = logging.getLogger('global')
    batch_size = len(image_info)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for b_ix in range(batch_size):
        image_size = image_info[b_ix]
        keep_ix = np.where(bboxes[:, 0] == b_ix)[0]
        bbs = bboxes[keep_ix]
        kps = keypoints[keep_ix, :, :2] if keypoints else None
        msks = [masks[ix] for ix in keep_ix] if masks else None

        hmap = heatmap[keep_ix]
        filename = debugger.get_filename(b_ix).split('/')[-1].split('.')[0]
        for r_ix, B in enumerate(bbs):
            box_score, class_id = B[-2:]
            if box_score < 0.9:
                continue

            image = debugger.get_image(b_ix).copy()
            x1, y1, x2, y2 = map(int, B[1:1+4])
            r_h = y2 - y1
            r_w = x2 - x1
            draw_bbox(image, B[1:1+4])
            category_name = class_names[int(class_id)]
            cv2.putText(image, 'category:{0}, score:{1}'.format(category_name,box_score), (100, 100), 2, 1, (0, 0, 255))
            logger.info('{0}/{1}_{2}.jpg'.format(results_dir, filename, r_ix))

            if kps:
                draw_keypoint(image, kps[r_ix])
                #for k in range(hmap.shape[1]):
                #    hp = hmap[r_ix, k]
                #    hp = cv2.resize(hp, (r_w, r_h)) * 250
                #    hp[hp < 0] = 0
                #    img = image.copy()
                #    img[y1:y2, x1:x2, ...] *= 0.5
                #    img[y1:y2, x1:x2, ...] += hp[..., np.newaxis] * 0.5
                #    cv2.imwrite('{0}/{1}_{2}_{3}.jpg'.format(results_dir, filename, r_ix, k), img)
                cv2.imwrite('{0}/{1}_{2}_keypoints.jpg'.format(results_dir, filename, r_ix), image)
                hp = cv2.resize(np.max(hmap[r_ix], axis=0), (r_w, r_h)) * 100
                hp[hp < 0] = 0
                image[y1:y2, x1:x2, ...] *= 0.5
                image[y1:y2, x1:x2, ...] += hp[..., np.newaxis] * 0.5
                cv2.imwrite('{0}/{1}_{2}_heatmap.jpg'.format(results_dir, filename, r_ix), image)
            if msks:
                draw_mask(image, msks[r_ix])
                cv2.imwrite('{0}/{1}_{2}_mask.jpg'.format(results_dir, filename, r_ix), image)

def vis_detections(img, bboxes, gts, img_name, score_thresh):
    vis_dir = 'visualize'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    img_name = img_name.rsplit('/',1)[-1].split('.')[0]
    overlaps = bbox_helper.bbox_iou_overlaps(bboxes, gts)
    max_overlaps = overlaps.max(axis=1)
    for box_ix in range(bboxes.shape[0]):
        box = bboxes[box_ix, :4].astype(np.int32)
        score = bboxes[box_ix, 4]
        if score < score_thresh:
            continue
        cls = int(bboxes[box_ix, 5])
        img_cpy = img.copy()
        ov = max_overlaps[box_ix]
        text = 'label:%s, iou:%.3f, score:%.3f' % (classes[cls], ov, score)
        cv2.putText(img_cpy, text, (30, 30), 2, 0.8, (0, 0, 255))
        vis = cv2.rectangle(img_cpy, tuple(box[0:2]), tuple(box[2:4]), (255, 0, 0))
        cv2.imwrite('%s/%s_%d.jpg' %(vis_dir, img_name, box_ix), vis)

def vis_batch(input, output_dir, prefix):
    from utils.debug_helper import debugger
    import logging
    logger = logging.getLogger('global')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if torch.is_tensor(input[0]):
        debugger.store_tensor_as_image(input[0])

    image_info = input[1]
    gt_boxes = input[2]
    ignores = input[3]
    kpts = input[4]
    masks = input[5]
    #filenames = input[6]
    B = gt_boxes.shape[0]
    for b in range(B):
        #image = imgs[b]
        image = debugger.get_image(b)
        bxs = gt_boxes[b]
        #igs = ignores[b]
        kts = kpts[b]
        #mks = masks[b]
        n = bxs.shape[0]
        for ix in range(n):
            img_cpy = image.copy()
            draw_bbox(img_cpy, bxs[ix])
            draw_keypoint(img_cpy, kts[ix])
            #draw_mask(img_cpy, mks[ix])
            filename = os.path.join(output_dir, '{0}_{1}_{2}.jpg'.format(prefix, b, ix))
            cv2.imwrite(filename, img_cpy)
        #for ix in range(igs.shape[0]):
        #    img_cpy = imgs[b].copy()
        #    draw_bbox(img_cpy, igs[ix], color=(0,0,255))
        #    filename = os.path.join(test_dir, '{0}_{1}_{2}.jpg'.format(prefix, b, ix + n))
        #    cv2.imwrite(filename, img_cpy)

def make_dot(var, params=None):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    if params is not None:
        # assert isinstance(params.values()[0], Variable)
        param_map = {id(v): k for k, v in params.items()}

    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="20,20"), format='svg')
    seen = set()

    def size_to_str(size):
        return '('+(', ').join(['%d' % v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                name = param_map[id(u)] if params is not None else ''
                node_name = '%s\n %s' % (name, size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot

def visualize(var, filename):
    make_dot()
