import torch
from extensions._bbox_helper._ext import bbox_helper
import numpy as np

def overlap(bboxes1, bboxes2):
    # bboxes1, bboxes2 has to be a tensor
    # bboxes1  [N, 4]: x1, y1, x2, y2
    # bboxes2  [M, 4]: x1, y1, x2, y2
    bboxes1 = torch.from_numpy(bboxes1[:, :4]).float().cuda().contiguous()
    bboxes2 = torch.from_numpy(bboxes2[:, :4]).float().cuda().contiguous()

    output = torch.cuda.FloatTensor(bboxes1.shape[0], bboxes2.shape[0])
    bbox_helper.gpu_iou_overlaps(bboxes1, bboxes2, output)

    return output.cpu().numpy()

