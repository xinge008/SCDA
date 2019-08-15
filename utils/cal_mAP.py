# -*- coding: utf-8 -*-
#-------------------------------------------
# cal mAP | base on pytorch example dataset
# for cityscapes specifically
# pang jiangmiao | 2018.04.15
#-------------------------------------------
# import sys
import numpy as np
from collections import defaultdict
import subprocess

# import pprint
# import pdb
import logging
logger = logging.getLogger('global')
def parse_gts(gts_list, num_classes):
    '''parse detection ground truths list
        dict[img_name] = {height:, width:, bbox_num:, bbox:{cls:[[x1,y1,x2,y2],...], ...} }
    '''
    logger.info('Start parsing gts list......')
    index_info = [temp for temp in enumerate(gts_list) if temp[1].startswith('#')]
    gts = defaultdict(list)
    gts['num'] = np.zeros(num_classes)
    for i in range(len(index_info)):
        index = index_info[i][0]
        img_name = gts_list[index + 1].strip()      # val/folder/img_name.png
        pure_name = img_name.split('/')[-1][0:-4]  # img_name
        gts[pure_name] = defaultdict(list)
        gts[pure_name]['height'] = gts_list[index+3].strip()
        gts[pure_name]['width'] = gts_list[index+4].strip()
        gts[pure_name]['bbox_num'] = int(gts_list[index+7])
        gts[pure_name]['bbox'] = defaultdict(list)
        for b in gts_list[index+8:index+8+int(gts_list[index+7])]:
            b = b.split()
            label = int(b[0])
            x1 = int(b[1])
            y1 = int(b[2])
            x2 = int(b[3])
            y2 = int(b[4])
            gts[pure_name]['bbox'][label].append([x1, y1, x2, y2])
            gts['num'][label] += 1
        gts[pure_name]['is_det'] = defaultdict(list)
        for l in range(1, num_classes):
            gts[pure_name]['is_det'][l] = np.zeros(len(gts[pure_name]['bbox'][l]))
    logger.info('Done!')
    return gts

def parse_res(res_list):
    '''parse results list
        dict[cls] = [[x1, y1, x2, y2, score, img_name], ...]
    '''
    logger.info('Start parsing results list......')
    results = defaultdict(list)
    for r in res_list:
        r = r.split()
        img_name = r[0]  # img_name no extension
        label = int(r[6])
        score = float(r[5])
        x1 = int(float(r[1]))
        y1 = int(float(r[2]))
        x2 = int(float(r[3]))
        y2 = int(float(r[4]))
        results[label].append([x1, y1, x2, y2, score, img_name])
    logger.info('Done!')
    return results

def calIoU(result, gt_i):
    # result: [x1, y1, x2, y2, score, img_name]
    # gts: [[x1, x2, y1, y2], []...]
    x1 = result[0]
    y1 = result[1]
    x2 = result[2]
    y2 = result[3]
    overmax = -1
    is_which = -1
    for k, gt in enumerate(gt_i):
        gt_x1 = gt[0]
        gt_y1 = gt[1]
        gt_x2 = gt[2]
        gt_y2 = gt[3]
        inter_x1 = max(x1, gt_x1)
        inter_y1 = max(y1, gt_y1)
        inter_x2 = min(x2, gt_x2)
        inter_y2 = min(y2, gt_y2)
        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            area_inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
            area_sum1 = (x2 - x1 + 1) * (y2 - y1 + 1)
            area_sum2 = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)
            IoU = area_inter/(area_sum1 + area_sum2 - area_inter)
            if IoU > overmax:
                overmax = IoU
                is_which = k
    return overmax, is_which

def cal_mAP(gts, results, num_classes, overlap_thre):
    ap = np.zeros(num_classes)
    max_recall = np.zeros(num_classes)
    for class_i in range(1, num_classes):
        results_i = results[class_i]
        res_num = len(results_i)
        tp = np.zeros(res_num)
        fp = np.zeros(res_num)
        sum_gt = gts['num'][class_i]
        logger.info('sum_gt: {}'.format(sum_gt))
        results_i = sorted(results_i, key = lambda xx : xx[4], reverse=True)
        for k, res in enumerate(results_i):
            img_name = res[-1]
            gts_i = gts[img_name]['bbox'][int(class_i)]
            overmax, is_which = calIoU(res, gts_i)
            if overmax >= overlap_thre and gts[img_name]['is_det'][class_i][is_which] == 0:
                tp[k] = 1
                gts[img_name]['is_det'][class_i][is_which] = 1
            else:
                fp[k] = 1
        rec = np.zeros(res_num)
        prec = np.zeros(res_num)
        for v in range(res_num):
            if v > 0:
                tp[v] = tp[v] + tp[v-1]
                fp[v] = fp[v] + fp[v-1]
            rec[v] = tp[v] / sum_gt
            prec[v] = tp[v] / (tp[v] + fp[v])
        for v in range(res_num-2, -1, -1):
            prec[v] = max(prec[v], prec[v+1])
        for v in range(res_num):
            if v == 0:
                ap[class_i] += rec[v] * prec[v]
            else:
                ap[class_i] += (rec[v] - rec[v-1]) * prec[v]
        max_recall[class_i] = np.max(rec)
        logger.info('class {} --- ap: {}   max recall: {}'.format(class_i, ap[class_i], max_recall[class_i]))
    return ap, max_recall


def Cal_MAP1(res_list, gts_list, num_classes):
    # with open(res_list, 'r') as f_res:
    #     res_list = f_res.readlines()
    # with open(gts_list, 'r') as f_gts:
    #     gts_list = f_gts.readlines()
    overlap_thre = 0.5
    num_classes = int(num_classes)
    gts = parse_gts(gts_list, num_classes)
    results = parse_res(res_list)

    ap, max_recall = cal_mAP(gts, results, num_classes, overlap_thre)
    mAP = np.mean(ap[1:])
    m_rec = np.mean(max_recall[1:])
    # print('--------------------')
    logger.info('mAP: {}   max recall: {}'.format(mAP, m_rec))
    # print('--------------------')
    return mAP

def Cal_MAP(res_dir, gts_list, num_classes):
    overlap_thre = 0.5
    res_list = 'results.txt'
    subprocess.call("cat {}/results.txt.rank* > {}/{}".format(res_dir,res_dir, res_list), shell=True)

    with open("{}/{}".format(res_dir, res_list), 'r', encoding='utf-8') as f_res:
        res_list = f_res.readlines()
    with open(gts_list, 'r', encoding='utf-8') as f_gts:
        gts_list = f_gts.readlines()

    gts = parse_gts(gts_list, num_classes)
    results = parse_res(res_list)

    ap, max_recall = cal_mAP(gts, results, num_classes, overlap_thre)
    mAP = np.mean(ap[1:])
    m_rec = np.mean(max_recall[1:])
    print('--------------------')
    print('mAP: {}   max recall: {}'.format(mAP, m_rec))
    print('--------------------')



