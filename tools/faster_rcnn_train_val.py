# -*- coding: utf-8 -*-
# @Time    : 18-7-15 11:33
# @Author  : Xinge

"""


"""

from __future__ import division
# workaround of the bug where 'import torchvision' sets the start method to be 'fork'
import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')

import cv2

cv2.ocl.setUseOpenCL(False)
import sys
import itertools

sys.setrecursionlimit(10000)
import subprocess
from datasets.coco_dataset import COCODataset, COCOTransform
from datasets.coco_loader import COCODataLoader
from datasets.example_dataset import ExampleDataset, ExampleTransform
from datasets.example_loader import ExampleDataLoader
from datasets.target_dataset import TargetDataset

from models.faster_rcnn.vgg_adver_expansion_cluster import vgg16 as vgg16_FasterRCNN
from models.faster_rcnn.vgg_adver_expansion_cluster import vgg16_bn as vgg16bn_FasterRCNN
from utils.cal_mAP import Cal_MAP

"""
for generator and discriminator in GAN
"""
from models.faster_rcnn.faster_rcnn_adver_expansion_reweight_cluster import GAN_dis_AE_patch, GAN_dis_AE, GAN_decoder_AE, GAN_decoder_AE_32

from utils import bbox_helper
# from utils.debug_helper import debugger
from utils.log_helper import init_log
from utils.lr_helper import IterExponentialLR
from utils.log_helper import print_speed
from utils.coco_eval import eval_coco_ap_from_results_txt as eval_coco_ap
from utils.load_helper import restore_from, load_pretrain

from utils.distributed_utils import dist_init, average_gradients, broadcast_params

from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torch.utils import data
import torch.nn.functional as F

import argparse
import functools
import logging
import os
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler
import json


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


model_zoo = [ 'vgg16_FasterRCNN', 'vgg16bn_FasterRCNN']

parser = argparse.ArgumentParser(description='PyTorch Detection Training')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('--dist', dest='dist', type=int, default=1,
                    help='distributed training or not')
parser.add_argument('--L1', dest='L1', type=int, default=0,
                    help='distributed training or not')
parser.add_argument('--fix_num', dest='fix_num', type=int, default=1,
                    help='number of conv layer')
parser.add_argument('--backend', dest='backend', type=str, default='nccl',
                    help='backend for distributed training')
parser.add_argument('--results_dir', dest='results_dir', default='results_dir',
                    help='results dir of output for each class')
parser.add_argument('--port', dest='port', required=True, help='port of server')
parser.add_argument('--save_dir', dest='save_dir', default='checkpoints',
                    help='directory to save models')
parser.add_argument('--warmup_epochs', dest='warmup_epochs', type=int, default=0,
                    help='epochs for warming up to enlarge lr')
parser.add_argument('--step_epochs', dest='step_epochs', type=lambda x: list(map(int, x.split(','))),
                    default='-1', help='epochs to decay lr')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of faster-rcnn in json format')
parser.add_argument('--arch', dest='arch', default='resnet101', choices=model_zoo,
                    help='architecture of pretrained model')
parser.add_argument('--ganmodel', dest='ganmodel', default=0, choices=[0, 1],
                    help='the path of pretrained GAN-model, 0 for vgg, 1 for resnet50')
parser.add_argument('--dataset', dest='dataset', required=True, choices=['pascal_voc', 'coco', 'cityscapes'],
                    help='which dataset is used for training')
parser.add_argument('--datadir', dest='datadir', required=True,
                    help='data directory of VOCdevkit2007 when dataset is `pascal_voc`,'
                         'otherwise, the image directory')
parser.add_argument('--train_meta_file', dest='train_meta_file', default='',
                    help='file for training which contains a list of image meta, including image path,'
                         'height, width, ground truth boxes, ignore bboxes, etc')
parser.add_argument('--val_meta_file', dest='val_meta_file', default='',
                    help='same as train_meta_file, but for validation')
parser.add_argument('--target_meta_file', dest='target_meta_file', default='',
                    help='same as target_meta_file, but for target')
parser.add_argument('--eval_interval', dest='eval_interval', type=int, default=1,
                    help='the interval of validation')
parser.add_argument('--new_w', dest='new_w', type=int, default=1024,
                    help='image size w')
parser.add_argument('--new_h', dest='new_h', type=int, default=512,
                    help='image size h')

parser.add_argument('--neww', dest='neww', type=int, default=64,
                    help='feature map size w')
parser.add_argument('--newh', dest='newh', type=int, default=64,
                    help='feature map size h')


#### for clusters######
"""
for some valid pairs:
cluster_num = 2, threshold = 256, recon_size = 512
cluster_num = 4, threshold = 128, recon_size = 256
cluster_num = 8, threshold = 64,  recon_size = 128
"""
parser.add_argument('--cluster_num', dest='cluster_num', type=int, default=4,
                    help='cluster number')
parser.add_argument('--threshold', dest='threshold', type=int, default=128,
                    help='proposals size of each cluster')
parser.add_argument('--recon_size', dest='recon_size', type=int, default=256,
                    help='reconstruction size')



"""
for optimizer part
"""

parser.add_argument('--sgdflag', dest='sgdflag', type=int, default=0, help='distributed training or not')

best_recall = 0
best_map = 0.0


def load_config(config_path):
    assert (os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    for key in cfg.keys():
        if key != 'shared':
            cfg[key].update(cfg['shared'])
    return cfg


def build_data_loader(dataset, cfg):
    logger = logging.getLogger('global')
    # Data loading code
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ospj = os.path.join
    if dataset == 'coco':
        logger.info("build coco dataset")
        Dataset = COCODataset
        DataLoader = COCODataLoader
        transform_fn = COCOTransform
        train_data_dir = os.path.join(args.datadir, "train2017")
        val_data_dir = os.path.join(args.datadir, "val2017")
        assert (args.train_meta_file != '')
        assert (args.val_meta_file != '')
    else:
        Dataset = ExampleDataset
        DataLoader = ExampleDataLoader
        transform_fn = ExampleTransform
        train_data_dir = val_data_dir = args.datadir
        assert (args.train_meta_file != '')
        assert (args.val_meta_file != '')

    scales = cfg['shared']['scales']
    max_size = cfg['shared']['max_size']
    logger.info('train meta file:{}'.format(args.train_meta_file))
    logger.info('val meta file:{}'.format(args.val_meta_file))
    train_dataset = Dataset(train_data_dir,
                            args.train_meta_file,
                            transform_fn(scales, max_size, flip=True),
                            normalize_fn=normalize)
    val_dataset = Dataset(val_data_dir,
                          args.val_meta_file,
                          transform_fn(max(scales), max_size, flip=False),
                          normalize_fn=normalize)

    target_dataset = TargetDataset(train_data_dir,
                                   args.target_meta_file,
                                   normalize_fn=normalize, new_h=args.new_h, new_w=args.new_w)

    logger.info('build dataset done')

    train_sampler = None
    val_sampler = None
    target_sampler = None
    if args.dist:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        target_sampler = DistributedSampler(target_dataset)

    logger.info('build dataset done')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False if train_sampler else True,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    target_loader = data.DataLoader(target_dataset, batch_size=args.batch_size,
                                    shuffle=False if target_sampler else True,
                                    num_workers=args.workers, pin_memory=False, sampler=target_sampler)

    return train_loader, val_loader, target_loader

def builder_gan(args):
    """
    hyper-params for GAN part
    """
    size2layers = {256:3, 512:4, 128:2}
    # size2genlayers = {256:}
    params_dec = {'ch': args.threshold, 'input_dim_a': 3, 'input_dim_b': 3, 'n_enc_front_blk': 5,
                  'n_enc_res_blk': 2, 'n_enc_shared_blk': 2, 'n_gen_shared_blk': 2,
                  'n_gen_res_blk': 3, 'n_gen_front_blk': size2layers[args.recon_size], 'res_dropout_ratio': 0.5, 'neww': args.neww,
                  'newh': args.newh, 'cluster_num': args.cluster_num, 'threshold': args.threshold}

    params_dis = {'input_dim_a': 3, 'input_dim_b': 3, 'ch': 32, 'n_gen_res_blk': 3, 'n_layer': size2layers[args.recon_size]}

    params_patch_dis = {'n_in': args.threshold, 'n_out': args.threshold * 2, 'cluster_num': args.cluster_num}

    dis_model = GAN_dis_AE(params_dis)
    dec_model = GAN_decoder_AE(params_dec)
    dis_model_patch = GAN_dis_AE_patch(params_patch_dis)

    return dis_model, dec_model, dis_model_patch

def main():
    init_log('global', logging.INFO)
    logger = logging.getLogger('global')
    global args, best_recall, best_map
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.dist:
        logger.info('dist:{}'.format(args.dist))
        dist_init(args.port, backend=args.backend)

    # build dataset
    train_loader, val_loader, target_loader = build_data_loader(args.dataset, cfg)


    if args.arch == 'vgg16_FasterRCNN':
        model = vgg16_FasterRCNN(pretrained=False, cfg=cfg['shared'])
    elif args.arch == 'vgg16bn_FasterRCNN':
        model = vgg16bn_FasterRCNN(pretrained=False, cfg=cfg['shared'])
    else:
        logger.info("The arch is not in model zoo")
        exit()
    logger.info('build model done')
    logger.info(model)

    if args.pretrained:
        model = load_pretrain(model, args.pretrained)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, args.lr, betas=(0.9, 0.999), weight_decay=0.0001)

    #=============build gan part===============
    dis_model, dec_model, dis_model_patch = builder_gan(args)
    trainable_params_dec = [p for p in dec_model.parameters() if p.requires_grad]
    trainable_params_dis = [p for p in dis_model.parameters() if p.requires_grad]
    trainable_params_dis_patch = [p for p in dis_model_patch.parameters() if p.requires_grad]
    dis_optimizer = torch.optim.Adam(trainable_params_dis, args.lr, betas=(0.9, 0.999),
                                     weight_decay=0.0001)
    dis_patch_optimizer = torch.optim.Adam(trainable_params_dis_patch, args.lr, betas=(0.9, 0.999),
                                     weight_decay=0.0001)
    dec_optimizer = torch.optim.Adam(trainable_params_dec, args.lr, betas=(0.9, 0.999), weight_decay=0.0001)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch, best_recall, arch = restore_from(model, optimizer, args.resume)

    model = model.cuda()
    dis_model = dis_model.cuda()
    dec_model = dec_model.cuda()
    dis_model_patch = dis_model_patch.cuda()
    if args.dist:
        # broadcast_params([model, dis_model, dec_model])
        broadcast_params(model)
        broadcast_params(dis_model)
        broadcast_params(dec_model)
        broadcast_params(dis_model_patch)

    logger.info('build dataloader done')
    if args.evaluate:
        if args.dist:
            rc = validate(val_loader, model, cfg)
            logger.info('recall=%f' % rc)
            return
        else:
            rc = validate_single(val_loader, model, cfg)
            logger.info('recall=%f' % rc)
            return

    # warmup to enlarge lr
    if args.start_epoch == 0 and args.warmup_epochs > 0:
        world_size = 1
        try:
            world_size = dist.get_world_size()
        except Exception as e:
            print(e)
        rate = world_size * args.batch_size
        warmup_iter = args.warmup_epochs * len(train_loader)
        assert (warmup_iter > 1)
        gamma = rate ** (1.0 / (warmup_iter - 1))
        lr_scheduler = IterExponentialLR(optimizer, gamma)
        lr_scheduler_dis = IterExponentialLR(dis_optimizer, gamma)
        lr_scheduler_dis_patch = IterExponentialLR(dis_patch_optimizer, gamma)

        lr_scheduler_dec = IterExponentialLR(dec_optimizer, gamma)
        for epoch in range(args.warmup_epochs):
            logger.info('warmup epoch %d' % (epoch))
            train(train_loader, target_loader, val_loader, model, dec_model, dis_model, dis_model_patch, lr_scheduler,
                  lr_scheduler_dec, lr_scheduler_dis, lr_scheduler_dis_patch, epoch + 1, cfg, warmup=True)
        # overwrite initial_lr with magnified lr through warmup
        for group in optimizer.param_groups + dis_optimizer.param_groups + dec_optimizer.param_groups + dis_patch_optimizer.param_groups:
            group['initial_lr'] = group['lr']
        logger.info('warmup for %d epochs done, start large batch training' % args.warmup_epochs)

    lr_scheduler = MultiStepLR(optimizer, milestones=args.step_epochs, gamma=0.1, last_epoch=args.start_epoch - 1)
    lr_scheduler_dis = MultiStepLR(dis_optimizer, milestones=args.step_epochs, gamma=0.1,
                                   last_epoch=args.start_epoch - 1)
    lr_scheduler_dec = MultiStepLR(dec_optimizer, milestones=args.step_epochs, gamma=0.1,
                                   last_epoch=args.start_epoch - 1)
    lr_scheduler_dis_patch = MultiStepLR(dis_patch_optimizer, milestones=args.step_epochs, gamma=0.1,
                                   last_epoch=args.start_epoch - 1)

    for epoch in range(args.start_epoch, args.epochs):
        logger.info('step_epochs:{}'.format(args.step_epochs))
        lr_scheduler.step()
        lr_scheduler_dis.step()
        lr_scheduler_dec.step()
        lr_scheduler_dis_patch.step()
        lr = lr_scheduler.get_lr()[0]
        # train for one epoch

        train(train_loader, target_loader, val_loader, model, dec_model, dis_model, dis_model_patch, lr_scheduler,
                lr_scheduler_dec, lr_scheduler_dis, lr_scheduler_dis_patch, epoch + 1, cfg)

        if (epoch + 1) % args.eval_interval == 0 or epoch + 1 == args.epochs:
            # evaluate on validation set
            recall = validate(val_loader, model, cfg)
            # remember best prec@1 and save checkpoint
            is_best = recall > best_recall
            best_recall = max(recall, best_recall)
            logger.info('recall %f(%f)' % (recall, best_recall))

        if (not args.dist) or (dist.get_rank() == 0):
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.cpu().state_dict(),
                'best_recall': best_recall,
                'optimizer': optimizer.state_dict(),
            }, False,
                os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch + 1)))


def get_corner_from_center(center):
    """
    :param center: (cluster_num, 2) > (center_x, center_y)
    :return: (cluster_num, 4) > (x1, y1, x2, y2)
    """
    corner = []
    recon_size = args.recon_size
    half_recon_size = recon_size // 2
    for cluster_idx in range(0, center.shape[0]):
        x_1 = max(int(center[cluster_idx][0]) - half_recon_size, 0)
        y_1 = max(int(center[cluster_idx][1]) - half_recon_size, 0)
        if x_1 == 0:
            x_2 = recon_size
        else:
            x_2 = min(int(center[cluster_idx][0]) + half_recon_size, args.new_w)  # for 1024
            if x_2 == args.new_w:
                x_1 = args.new_w - recon_size

        if y_1 == 0:
            y_2 = recon_size
        else:
            y_2 = min(int(center[cluster_idx][1]) + half_recon_size, args.new_h)  # for 512
            if y_2 == args.new_h:
                y_1 = args.new_h - recon_size
        corner.append([x_1, y_1, x_2, y_2])

    # assert ((x_2 - x_1) == 256),
    return corner

def generate_soft_label(flag, refer_tensor):
    if flag==1:
        labels = torch.from_numpy(np.random.uniform(0.8, 1.0, size=refer_tensor.size())).float().cuda()
    elif flag == 0:
        labels = torch.from_numpy(np.random.uniform(0.0, 0.3, size=refer_tensor.size())).float().cuda()
    else:
        raise Exception('unknown soft label Type')

    return labels

def generate_hard_label(flag, refer_tensor):
    if flag==1:
        labels = torch.from_numpy(np.random.uniform(1.0, 1.0, size=refer_tensor.size())).float().cuda()
    elif flag == 0:
        labels = torch.from_numpy(np.random.uniform(0.0, 0.0, size=refer_tensor.size())).float().cuda()
    else:
        raise Exception('unknown hard label Type')

    return labels


def train(train_loader, target_loader, val_loader, model, dec_model, dis_model, dis_model_patch, lr_scheduler,
          lr_scheduler_dec, lr_scheduler_dis, lr_scheduler_dis_patch, epoch, cfg, warmup=False):
    logger = logging.getLogger('global')
    model.cuda()
    model.train()
    dis_model.cuda()
    dis_model.train()
    dec_model.cuda()
    dec_model.train()
    dis_model_patch.cuda()
    dis_model_patch.train()

    if args.dist:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        world_size = 1
        rank = 0

    def freeze_bn(m):
        classname = m.__class__.__name__
        if classname.find('Norm') != -1:
            m.eval()

    model.apply(freeze_bn)
    fix_num = args.fix_num
    count = 1
    for mm in model.modules():
        if count > fix_num:
            break
        if isinstance(mm, torch.nn.Conv2d) and count <= fix_num:
            mm.eval()
            count += 1

    # dec_model.apply(freeze_bn)
    logger.info('freeze bn')

    end = time.time()

    t0 = time.time()
    l1_loss = torch.nn.L1Loss()
    if args.dist:
        # update random seed
        train_loader.sampler.set_epoch(epoch)
        target_loader.sampler.set_epoch(epoch)

    for iter, (input, target) in enumerate(zip(train_loader, target_loader)):
        # torch.cuda.empty_cache()
        if warmup:
            # update lr for each iteration
            lr_scheduler.step()
            lr_scheduler_dis.step()
            lr_scheduler_dec.step()
            lr_scheduler_dis_patch.step()
        x = {
            'cfg': cfg,
            'image': (input[0]).cuda(),
            'image_info': input[1],
            'ground_truth_bboxes': input[2],
            'ignore_regions': None,
            'cluster_num': args.cluster_num,
            'threshold': args.threshold
            # 'ignore_regions': input[3] if args.dataset == 'coco' else None
        }
        target = (target).cuda()
        outputs = model(x, target)

        centers_source, centers_target = outputs['cluster_centers']

        corners_source = get_corner_from_center(centers_source)
        corners_target = get_corner_from_center(centers_target)

        x_small = []
        target_small = []
        for corners_idx in range(0, len(corners_source)):
            x1 = corners_source[corners_idx][0]
            y1 = corners_source[corners_idx][1]
            x2 = corners_source[corners_idx][2]
            y2 = corners_source[corners_idx][3]
            assert (x2 - x1 == args.recon_size), "x size does not match 256 in source "
            assert (y2 - y1 == args.recon_size), "y size does not match 256 in source "
            x_small_tmp = x['image'][:, :, y1:y2, x1:x2]
            x_small.append(x_small_tmp)

        x_small = torch.cat(x_small, 0)

        for corners_idx in range(0, len(corners_target)):
            x1 = corners_target[corners_idx][0]
            y1 = corners_target[corners_idx][1]
            x2 = corners_target[corners_idx][2]
            y2 = corners_target[corners_idx][3]
            assert (x2 - x1 == args.recon_size), "x size does not match 256 in target "
            assert (y2 - y1 == args.recon_size), "y size does not match 256 in target "
            target_small_tmp = target[:, :, y1:y2, x1:x2]
            target_small.append(target_small_tmp)

        target_small = torch.cat(target_small, 0)  # Size(4, 3, 256, 256)

        x_source_patch, x_target_patch = outputs['cluster_features']  # Size(4, 128, 4096)
        x_source_recon, x_target_recon = dec_model(x_source_patch, x_target_patch)  # Size(4, 3, 256, 256)


        ##########################################################################
        ######################### (1): start dis_update ##########################
        ##########################################################################

        lr_scheduler_dis.optimizer.zero_grad()
        x_source_dis, x_target_dis = dis_model(x_source_recon, x_target_recon) # (4, 256)
        x_source_real, x_target_real = dis_model(x_small, target_small) # (4, 256)
        x_source_dis = torch.sigmoid(x_source_dis)  # (4, dim)
        x_target_dis = torch.sigmoid(x_target_dis)  # (4, dim)
        x_source_real = torch.sigmoid(x_source_real)  # (4, dim)
        x_target_real = torch.sigmoid(x_target_real)

        x_source_dis_cluster = torch.split(x_source_dis, 1, dim=0)
        x_source_real_cluster = torch.split(x_source_real, 1, dim=0)
        score_1_cluster = generate_soft_label(1, x_source_real_cluster[0])
        score_0_cluster = generate_soft_label(0, x_source_dis_cluster[0])

        adloss_source = 0.0

        #################### (1.1): for source clusters############################
        for clu_idx in range(0, len(x_source_dis_cluster)):
            adloss_source +=  (F.binary_cross_entropy(x_source_dis_cluster[clu_idx], score_1_cluster) +
                              F.binary_cross_entropy(x_source_real_cluster[clu_idx], score_0_cluster))


        #################### (1.2): for target clusters############################

        x_target_patch_pro = dis_model_patch(x_target_patch)
        x_target_patch_pro_mean = torch.mean(x_target_patch_pro, 1)  # Size(4,1)
        x_source_patch_pro = dis_model_patch(x_source_patch)  # (4, 512)

        x_target_dis_cluster = torch.split(x_target_dis, 1, dim=0)
        x_target_real_cluster = torch.split(x_target_real, 1, dim=0)

        adloss_target = 0.0
        for clu_idx in range(0, len(x_target_dis_cluster)):
            adloss_target +=  (x_target_patch_pro_mean[clu_idx] * F.binary_cross_entropy(x_target_dis_cluster[clu_idx], score_0_cluster) +
                              F.binary_cross_entropy(x_target_real_cluster[clu_idx], score_1_cluster))



        adloss = (adloss_source + adloss_target) / world_size
        adloss.backward(retain_graph=True)

        max_grad3 = 0.0
        for pp in dis_model.parameters():
            tmp = torch.max(pp.grad.data)
            if max_grad3 < tmp:
                max_grad3 = tmp

        if args.dist:
            average_gradients(dis_model)

        lr_scheduler_dis.optimizer.step()


        ##########################################################################
        ####################### (2): start dis_patch_update ######################
        ##########################################################################

        lr_scheduler_dis_patch.optimizer.zero_grad()
        score_0_patch = generate_soft_label(0, x_target_patch_pro)
        score_1_patch = generate_soft_label(1, x_source_patch_pro)

        patch_loss_target = F.binary_cross_entropy(x_target_patch_pro, score_0_patch)
        patch_loss_source = F.binary_cross_entropy(x_source_patch_pro, score_1_patch)

        dis_patch_loss =  (patch_loss_source + patch_loss_target) / world_size
        dis_patch_loss.backward(retain_graph = True)
        if args.dist:
            average_gradients(dis_model_patch)

        lr_scheduler_dis_patch.optimizer.step()


        ##########################################################################
        ########################## (3): start decoder_update #####################
        ##########################################################################

        lr_scheduler_dec.optimizer.zero_grad()


        # x_source_recon, x_target_recon = dec_model(x_target_patch, x_source_patch)
        x_source_dis, x_target_dis = dis_model(x_source_recon, x_target_recon)
        x_source_dis = torch.sigmoid(x_source_dis)  # (4, dim)

        x_source_real, x_target_real = dis_model(x_small, target_small) # (4, 256)
        x_source_real = torch.sigmoid(x_source_real)
        x_target_real = torch.sigmoid(x_target_real)


        """
        for the patch loss of Target image
        """
        x_target_patch_pro = dis_model_patch(x_target_patch)  # size(4, dim)
        gtav_dis_sigmoid_target = torch.sigmoid(x_target_dis)



        """
        obtain the weighting factor of target patches and calculate the target loss
        """
        x_target_patch_pro_mean2 = torch.mean(x_target_patch_pro, 1)  # Size(4,1)
        fake_loss1_target = 0.0
        gtav_dis_sigmoid_target = torch.split(gtav_dis_sigmoid_target, 1, dim=0)
        # allone_target_1 = (torch.ones(gtav_dis_sigmoid_target[0].size()).float().cuda())
        all_target_1 = generate_hard_label(1, gtav_dis_sigmoid_target[0])

        gtav_real_sigmoid_target = torch.split(x_target_real, 1, dim=0)
        all_target_0 = generate_hard_label(0, gtav_real_sigmoid_target[0])

        for clu_idx in range(0, len(gtav_dis_sigmoid_target)):
            fake_loss1_target += x_target_patch_pro_mean2[clu_idx] * (F.binary_cross_entropy(gtav_dis_sigmoid_target[clu_idx], all_target_1)+
                                                                      F.binary_cross_entropy(gtav_real_sigmoid_target[clu_idx], all_target_0))


        fake_loss1_source = 0.0
        x_source_fake_cluster2 = torch.split(x_source_dis, 1, dim=0)
        all_source_1 = generate_hard_label(1, x_source_fake_cluster2[0])
        x_source_real_cluster2 = torch.split(x_source_real, 1, dim=0)
        all_source_0 = generate_hard_label(0, x_source_real_cluster2[0])

        for clu_idx in range(0, len(x_source_fake_cluster2)):
            fake_loss1_source +=  (F.binary_cross_entropy(x_source_fake_cluster2[clu_idx], all_source_1) +
                                   F.binary_cross_entropy(x_source_real_cluster2[clu_idx], all_source_0))


        recon_loss = (fake_loss1_source + fake_loss1_target ) / world_size  # no-discriminator in the Decoder

        # recon_loss = recon_loss
        recon_loss.backward(retain_graph=True)

        max_grad2 = 0.0
        for pp in dec_model.parameters():
            tmp = torch.max(pp.grad.data)
            if max_grad2 < tmp:
                max_grad2 = tmp

        if args.dist:
            average_gradients(dec_model)
        # torch.nn.utils.clip_grad_norm(dec_model.parameters(), 10.0)
        lr_scheduler_dec.optimizer.step()


        ##########################################################################
        ########################### (4): start detection_update ##################
        ##########################################################################


        """
        target feature maps --> source reconstruction
        for cross-domain alignment
        """
        x_source_recon, x_target_recon = dec_model(x_target_patch, x_source_patch)
        x_source_dis, x_target_dis = dis_model(x_source_recon, x_target_recon)  # (4, dim)
        """
        weight of target patches
        """
        x_fake_dis_sigmoid = torch.sigmoid(x_target_dis) #
        allone_11 = generate_hard_label(1, x_fake_dis_sigmoid)
        fake_loss_source = F.binary_cross_entropy(x_fake_dis_sigmoid, allone_11)  # NO discriminator in Detection
        x_fake_dis_sigmoid2 = torch.sigmoid(x_source_dis)

        x_fake_dis_sigmoid2_cluster = torch.split(x_fake_dis_sigmoid2, 1, dim=0)
        allone_11_cluster = (torch.ones(x_fake_dis_sigmoid2_cluster[0].size()).float().cuda())


        fake_loss_target = 0.0
        for clu_idx in range(0, len(x_fake_dis_sigmoid2_cluster)):
            fake_loss_target += x_target_patch_pro_mean2[clu_idx] * (F.binary_cross_entropy(x_fake_dis_sigmoid2_cluster[clu_idx],allone_11_cluster))

        rpn_cls_loss, rpn_loc_loss, rcnn_cls_loss, rcnn_loc_loss = outputs['losses']
        # gradient is averaged by normalizing the loss with world_size
        loss = (rpn_cls_loss + rpn_loc_loss + rcnn_cls_loss + rcnn_loc_loss + 0.1 * (fake_loss_source +  fake_loss_target)) / world_size

        lr_scheduler.optimizer.zero_grad()
        loss.backward()

        max_grad1 = 0.0
        for pp in model.parameters():
            tmp = torch.max(pp.grad.data)
            if max_grad1 < tmp:
                max_grad1 = tmp

        if args.dist:
            average_gradients(model)
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        lr_scheduler.optimizer.step()


        ##########################################################################
        ################################ Output information ######################
        ##########################################################################

        rpn_accuracy = outputs['accuracy'][0][0] / 100.
        rcnn_accuracy = outputs['accuracy'][1][0] / 100.

        t2 = time.time()
        lr = lr_scheduler.get_lr()[0]
        logger.info('Epoch: [%d][%d/%d] LR:%f Time: %.3f Loss: %.5f (rpn_cls: %.5f rpn_loc: %.5f rpn_acc: %.5f'
                    ' rcnn_cls: %.5f, rcnn_loc: %.5f rcnn_acc:%.5f fake_loss: %.5f dec_loss: %.5f  dis_loss: %.5f fake_loss1: %.5f)' %
                    (epoch, iter, len(train_loader), lr, t2 - t0, loss.item() * world_size,
                     rpn_cls_loss.item(), rpn_loc_loss.item(), rpn_accuracy,
                     rcnn_cls_loss.item(), rcnn_loc_loss.item(), rcnn_accuracy, fake_loss_target.item(), recon_loss.item(),
                     adloss.item(), fake_loss1_source.item()))
        print_speed((epoch - 1) * len(train_loader) + iter + 1, t2 - t0, args.epochs * len(train_loader))
        t0 = t2
        logger.info("Max Grad, Det: %5f, Dec: %5f, Dis: %5f" % (max_grad1, max_grad2, max_grad3))


def validate(val_loader, model, cfg):
    global best_map
    logger = logging.getLogger('global')
    try:
        rank = dist.get_rank()
        logger.info("rank %d" % rank)
        world_size = dist.get_world_size()
    except Exception as e:
        print(e)
        rank, world_size = 0, 1

    # switch to evaluate mode
    model.eval()

    total_rc = 0
    total_gt = 0

    logger.info('start validate')
    if not os.path.exists(args.results_dir):
        try:
            os.makedirs(args.results_dir)
        except Exception as e:
            print(e)
    # remove the original results file
    if rank == 0:
        for f in os.listdir(args.results_dir):
            if 'results.txt.rank' in f and int(f.split('k')[-1]) >= world_size:
                logger.info("remove %s" % f)
                os.remove(os.path.join(args.results_dir, f))

    fout = open(os.path.join(args.results_dir, 'results.txt.rank%d' % rank), 'w')

    for iter, input in enumerate(val_loader):
        img = (input[0]).cuda()
        img_info = input[1]
        gt_boxes = input[2]
        filenames = input[-1]
        x = {
            'cfg': cfg,
            'image': img,
            'image_info': img_info,
            'ground_truth_bboxes': gt_boxes,
            'ignore_regions': None}
        batch_size = img.shape[0]
        t1 = time.time()
        t0 = time.time()
        outputs = model(x)['predict']
        t2 = time.time()

        proposals = outputs[0].data.cpu().numpy()
        bboxes = outputs[1].data.cpu().numpy()
        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu().numpy()
        for b_ix in range(batch_size):
            img_id = filenames[b_ix].rsplit('/', 1)[-1].rsplit('.', 1)[0]
            img_resize_scale = img_info[b_ix, -1]
            if args.dataset == 'coco':
                img_resize_scale = img_info[b_ix, 2]
            rois_per_image = proposals[proposals[:, 0] == b_ix]
            dts_per_image = bboxes[bboxes[:, 0] == b_ix]
            gts_per_image = gt_boxes[b_ix]
            # rpn recall
            num_rc, num_gt = bbox_helper.compute_recall(rois_per_image[:, 1:1 + 4], gts_per_image)
            total_gt += num_gt
            total_rc += num_rc
            order = dts_per_image[:, -2].argsort()[::-1][:100]
            dts_per_image = dts_per_image[order]

            # faster-rcnn eval
            for cls in range(1, cfg['shared']['num_classes']):
                dts_per_cls = dts_per_image[dts_per_image[:, -1] == cls]
                gts_per_cls = gts_per_image[gts_per_image[:, -1] == cls]
                dts_per_cls = dts_per_cls[:, 1:-1]
                # dts_per_cls = dts_per_cls[dts_per_cls[:, -1] > 0.05]
                gts_per_cls = gts_per_cls[:, :-1]
                dts_per_cls = bbox_helper.clip_bbox(dts_per_cls, img_info[b_ix, :2])
                if len(dts_per_cls) > 0:
                    dts_per_cls[:, :4] = dts_per_cls[:, :4] / img_resize_scale
                if len(gts_per_cls) > 0:
                    gts_per_cls[:, :4] = gts_per_cls[:, :4] / img_resize_scale
                for bx in dts_per_cls:
                    if args.dataset == 'coco':
                        fout.write('val2017/{0}.jpg {1} {2}\n'.format(img_id, ' '.join(map(str, bx)), cls))
                    else:
                        fout.write('{0} {1} {2}\n'.format(img_id, ' '.join(map(str, bx)), cls))
                fout.flush()
        logger.info('Test: [%d/%d] Time: %.3f %d/%d' % (iter, len(val_loader), t2 - t0, total_rc, total_gt))
        print_speed(iter + 1, t2 - t0, len(val_loader))
    logger.info('rpn300 recall=%f' % (total_rc / total_gt))
    fout.close()

    """
    eval the cityscapes for getting the map
    """

    # eval coco ap with official python api
    if args.dataset == 'coco':
        # sync all gpu for results has been writen done
        sync_tensor = torch.cuda.FloatTensor(1)
        dist.all_reduce(sync_tensor, op=dist.reduce_op.SUM)
        logger.info("sync over: {}".format(sync_tensor[0]))
        if rank == 0:
            logger.info('start eval coco mAP with official api ...')
            eval_coco_ap(args.results_dir, 'bbox', args.val_meta_file)
    else:
        sync_tensor = torch.cuda.FloatTensor(1)
        dist.all_reduce(sync_tensor, op=dist.reduce_op.SUM)
        logger.info("sync over: {}".format(sync_tensor[0]))
        if rank == 0:
            Cal_MAP(args.results_dir, args.val_meta_file, int(cfg['shared']['num_classes']))

    return total_rc / total_gt

def validate_single(val_loader, model, cfg):
    global best_map
    logger = logging.getLogger('global')

    rank, world_size = 0, 1

    # switch to evaluate mode
    model.eval()
    total_rc = 0
    total_gt = 0

    logger.info('start validate')
    if not os.path.exists(args.results_dir):
        try:
            os.makedirs(args.results_dir)
        except Exception as e:
            print(e)
    # remove the original results file
    # if rank == 0:
    for f in os.listdir(args.results_dir):
        if 'results.txt.rank' in f and int(f.split('k')[-1]) >= world_size:
            logger.info("remove %s" % f)
            os.remove(os.path.join(args.results_dir, f))

    fout = open(os.path.join(args.results_dir, 'results.txt.rank%d' % rank), 'w')

    for iter, input in enumerate(val_loader):
        img = (input[0]).cuda()
        img_info = input[1]
        gt_boxes = input[2]
        filenames = input[-1]
        x = {
            'cfg': cfg,
            'image': img,
            'image_info': img_info,
            'ground_truth_bboxes': gt_boxes,
            'ignore_regions': None}
        batch_size = img.shape[0]
        t1 = time.time()
        t0 = time.time()
        outputs = model(x)['predict']
        t2 = time.time()

        proposals = outputs[0].data.cpu().numpy()
        bboxes = outputs[1].data.cpu().numpy()
        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu().numpy()
        for b_ix in range(batch_size):
            img_id = filenames[b_ix].rsplit('/', 1)[-1].rsplit('.', 1)[0]
            img_resize_scale = img_info[b_ix, -1]
            if args.dataset == 'coco':
                img_resize_scale = img_info[b_ix, 2]
            rois_per_image = proposals[proposals[:, 0] == b_ix]
            dts_per_image = bboxes[bboxes[:, 0] == b_ix]
            gts_per_image = gt_boxes[b_ix]
            # rpn recall
            num_rc, num_gt = bbox_helper.compute_recall(rois_per_image[:, 1:1 + 4], gts_per_image)
            total_gt += num_gt
            total_rc += num_rc
            order = dts_per_image[:, -2].argsort()[::-1][:100]
            dts_per_image = dts_per_image[order]

            # faster-rcnn eval
            for cls in range(1, cfg['shared']['num_classes']):
                dts_per_cls = dts_per_image[dts_per_image[:, -1] == cls]
                gts_per_cls = gts_per_image[gts_per_image[:, -1] == cls]
                dts_per_cls = dts_per_cls[:, 1:-1]
                # dts_per_cls = dts_per_cls[dts_per_cls[:, -1] > 0.05]
                gts_per_cls = gts_per_cls[:, :-1]
                dts_per_cls = bbox_helper.clip_bbox(dts_per_cls, img_info[b_ix, :2])
                if len(dts_per_cls) > 0:
                    dts_per_cls[:, :4] = dts_per_cls[:, :4] / img_resize_scale
                if len(gts_per_cls) > 0:
                    gts_per_cls[:, :4] = gts_per_cls[:, :4] / img_resize_scale
                for bx in dts_per_cls:
                    if args.dataset == 'coco':
                        fout.write('val2017/{0}.jpg {1} {2}\n'.format(img_id, ' '.join(map(str, bx)), cls))
                    else:
                        fout.write('{0} {1} {2}\n'.format(img_id, ' '.join(map(str, bx)), cls))
                fout.flush()
        logger.info('Test: [%d/%d] Time: %.3f %d/%d' % (iter, len(val_loader), t2 - t0, total_rc, total_gt))
        print_speed(iter + 1, t2 - t0, len(val_loader))
    logger.info('rpn300 recall=%f' % (total_rc / total_gt))
    fout.close()

    """
    eval the cityscapes for getting the map
    """

    # eval coco ap with official python api
    if args.dataset == 'coco':
        eval_coco_ap(args.results_dir, 'bbox', args.val_meta_file)
    else:
        Cal_MAP(args.results_dir, args.val_meta_file, int(cfg['shared']['num_classes']))

    return total_rc / total_gt

def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    # if is_best:
    #    shutil.copyfile(filename, 'model_best.pth')


def adjust_learning_rate(optimizer, rate, gradual=True):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = None
    for param_group in optimizer.param_groups:
        if gradual:
            param_group['lr'] *= rate
        else:
            param_group['lr'] = args.lr * rate
        lr = param_group['lr']
    return lr


if __name__ == '__main__':
    main()
