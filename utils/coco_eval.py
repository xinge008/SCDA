from __future__ import division

from datasets.pycocotools.coco import COCO
from datasets.pycocotools.cocoeval import COCOeval
from datasets.coco_dataset import COCODataset
import sys,os
import re
import logging

logger = logging.getLogger('global')
def eval_coco_ap_from_results_txt(result_dir, test_type, anno_file):
    logger.info("start eval coco ...")

    assert(test_type in ['segm', 'bbox', 'keypoints', 'person_bbox', 'person_proposal', 'proposal'])
   
    category_ids = set()
    coco_gt = COCO(anno_file)
    for anno in coco_gt.anns.values():
        category_ids.add(anno['category_id'])
    class_to_category = {i+1:c for i, c in enumerate(sorted(category_ids))}
    
    all_res = []
    for f in os.listdir(result_dir):
        if 'results.txt.rank' in f:
            for aline in open(os.path.join(result_dir, f),'r'):
                aline = aline.rstrip().split()
                res = {}
                res["image_id"] = int(re.split('[/.]', aline[0])[-2])
                x1 = float(aline[1])
                y1 = float(aline[2])
                x2 = float(aline[3])
                y2 = float(aline[4])
                if test_type == 'proposal':
                    res["bbox"] = [x1, y1, x2, y2]
                    res["score"]= float(aline[-1])
                    res["category_id"] = 1
                else:
                    res["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                    res["score"]= float(aline[-2])
                    res["category_id"] = class_to_category[int(aline[-1])]
                all_res.append(res)

    logger.info("all res line: {}".format(len(all_res)))
    
    #prefix = {'keypoints':'person_keypoints', 'person_bbox':'person_keypoints',
    #        'bbox':'instances', 'segm':'instances',
    #        'proposal': 'instances', 'person_proposal':'person_keypoints'}[test_type]
    iou_type = {'keypoints':'keypoints', 'person_bbox':'bbox',
            'bbox':'bbox', 'segm':'segm',
            'proposal': 'bbox', 'person_proposal':'bbox'}[test_type]

    logger.info('loading annotations from %s\n' % anno_file)
    coco_dt = coco_gt.loadRes(all_res)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)

    if test_type.find('proposal') >= 0:
        coco_eval.params.useCats = 0
        coco_eval.params.maxDets = [1,100,1000]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

