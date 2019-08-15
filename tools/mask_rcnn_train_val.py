from __future__ import division
# workaround of the bug where 'import torchvision' sets the start method to be 'fork'
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')

from datasets.coco_dataset import COCODataset, COCOTransform
from datasets.coco_loader import COCODataLoader
from datasets.example_dataset import ExampleDataset, ExampleTransform
from datasets.example_loader import ExampleDataLoader
# from datasets.sampler import AspectRatioSortedDistributedSampler

from models.mask_rcnn import resnet


from utils import bbox_helper
from utils.log_helper import init_log
from utils.lr_helper import IterExponentialLR
from utils.log_helper import print_speed
from utils.load_helper import restore_from, load_pretrain

from utils.distributed_utils import dist_init, average_gradients, broadcast_params

from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn

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
import cv2
from PIL import Image

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default='',
                    help='use pre-trained model')
parser.add_argument('--dist', dest='dist', type=int, default=1,
                    help='distributed training or not')
parser.add_argument('--backend', dest='backend', type=str, default='nccl',
                    help='backend for distributed training')
parser.add_argument('--results_dir', dest='results_dir', default='results_dir',
                    help='results dir of output for each class')
parser.add_argument('--port', dest='port', required=True,
                    help='port of server')
parser.add_argument('--save_dir', dest='save_dir', default='checkpoints',
                    help='directory to save models')
parser.add_argument('--warmup_epochs', dest='warmup_epochs', type=int, default=0,
                    help='epochs for warming up to enlarge lr')
parser.add_argument('--step_epochs', dest='step_epochs', type=lambda x: list(map(int, x.split(','))),
                    default='-1', help='epochs to decay lr')
parser.add_argument('--config', dest='config', required=True,
                    help='hyperparameter of faster-rcnn in json format')
parser.add_argument('--arch', dest='arch', default='resnet50',
                    help = 'architecture of pretrained model, currently only ResNet is supported')
parser.add_argument('--dataset', dest='dataset', required=True, choices = ['pascal_voc', 'coco'],
                    help='which dataset is used for training')
parser.add_argument('--datadir', dest='datadir', required=True,
                    help='data directory of VOCdevkit2007 when dataset is `pascal_voc`,'
                    'otherwise, the image directory')
parser.add_argument('--train_meta_file', dest='train_meta_file', default = '',
                    help = 'file for training which contains a list of image meta, including image path,'
                    'height, width, ground truth boxes, ignore bboxes, etc')
parser.add_argument('--val_meta_file', dest='val_meta_file', default = '',
                    help = 'same as train_meta_file, but for validation')
parser.add_argument('--has-keypoint', action='store_true',
                    help='has keypoints or not')
parser.add_argument('--has-mask', action='store_true',
                    help='has mask or not')

best_recall = 0

def load_config(config_path):
    assert(os.path.exists(config_path))
    cfg = json.load(open(config_path, 'r'))
    logger = logging.getLogger('global')
    logger.info(json.dumps(cfg, indent=2))
    for key in cfg.keys():
        if key != 'shared':
            cfg[key].update(cfg['shared'])
    return cfg

def build_data_loader(dataset, cfg):
    logger = logging.getLogger('global')
    # Data loading code
    normalize = transforms.Normalize(mean=[0.4485295, 0.4249905, 0.39198247],
                                     std=[0.12032582, 0.12394787, 0.14252729])
    ospj = os.path.join

    if dataset == 'coco':
        logger.info("build coco dataset")
        Dataset = COCODataset
        DataLoader = COCODataLoader
        transform_fn = COCOTransform
        assert (args.train_meta_file != '')
        assert (args.val_meta_file != '')
    else:
        Dataset = ExampleDataset
        DataLoader = ExampleDataLoader
        transform_fn = ExampleTransform
        assert (args.train_meta_file != '')
        assert (args.val_meta_file != '')

    scales = cfg['shared']['scales']
    max_size = cfg['shared']['max_size']
    logger.info('train meta file:{}'.format(args.train_meta_file))
    logger.info('val meta file:{}'.format(args.val_meta_file))
    train_dataset = Dataset(os.path.join(args.datadir, 'train2017'),
                            args.train_meta_file,
                            transform_fn(scales, max_size, flip=True),
                            normalize_fn=normalize,
                            has_keypoint=args.has_keypoint,
                            has_mask=args.has_mask)
    val_dataset = Dataset(os.path.join(args.datadir, 'val2017'),
                          args.val_meta_file,
                          transform_fn(max(scales), max_size, flip=False),
                          normalize_fn=normalize,
                          has_keypoint=args.has_keypoint,
                          has_mask=args.has_mask)
    logger.info('build dataset done')

    train_sampler = None
    val_sampler = None
    if args.dist:
        # AspectRatioSortedDistributedSampler will sort images by their aspect ratio after sampling.
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)

    logger.info('build dataset done')
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False if train_sampler else True,
        num_workers=args.workers, pin_memory=False, sampler=train_sampler)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=False, sampler=val_sampler)
    return train_loader, val_loader




def main():
    init_log('global', logging.INFO)
    logger = logging.getLogger('global')
    global args, best_recall
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.dist:
        logger.info('dist:{}'.format(args.dist))
        dist_init(args.port, backend = args.backend)

    # build dataset
    train_loader, val_loader = build_data_loader(args.dataset, cfg)
    # if args.arch == 'resnext_101_64x4d_deform_maskrcnn':
    #     model = resnext_101_64x4d_deform_maskrcnn(cfg = cfg['shared'])
    # elif args.arch == 'FishMask':
    #     model = FishMask(cfg = cfg['shared'])
    # else:
    #     if args.arch.find('fpn'):
    #         arch = args.arch.replace('fpn', '')
    #         model = resnet_fpn.__dict__[arch](pretrained=False, cfg = cfg['shared'])
    #     else:
    model = resnet.__dict__[args.arch](pretrained=False, cfg = cfg['shared'])
    logger.info('build model done')
    logger.info(model)

    if args.pretrained:
        model = load_pretrain(model, args.pretrained)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch, best_recall, arch = restore_from(model, optimizer, args.resume)

    model = model.cuda()
    if args.dist:
        broadcast_params(model)

    logger.info('build dataloader done')
    if args.evaluate:
        rc = validate(val_loader, model, cfg)
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
        assert(warmup_iter > 1)
        gamma = rate ** (1.0 / (warmup_iter-1))
        lr_scheduler = IterExponentialLR(optimizer, gamma)
        for epoch in range(args.warmup_epochs):
            logger.info('warmup epoch %d' % (epoch))
            train(train_loader, model, lr_scheduler, epoch + 1, cfg, warmup=True)
        # overwrite initial_lr with magnified lr through warmup
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
        logger.info('warmup for %d epochs done, start large batch training' % args.warmup_epochs)

    lr_scheduler = MultiStepLR(optimizer, milestones = args.step_epochs, gamma = 0.1, last_epoch = args.start_epoch-1)
    for epoch in range(args.start_epoch, args.epochs):
        logger.info('step_epochs:{}'.format(args.step_epochs))
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()[0]
        # train for one epoch
        train(train_loader, model, lr_scheduler, epoch + 1, cfg)

        if (epoch+1) % 5 == 0 or epoch+1 == args.epochs:
            # evaluate on validation set
            recall = validate(val_loader, model, cfg)
            # remember best prec@1 and save checkpoint
            is_best = recall > best_recall
            best_recall = max(recall, best_recall)
            logger.info('recall %f(%f)' % (recall, best_recall))

        if (not args.dist) or (dist.get_rank() == 0):
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch + 1))
            torch.save({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.cpu().state_dict(),
                'best_recall': best_recall,
                'optimizer': optimizer.state_dict(),
            }, save_path)


def train(train_loader, model, lr_scheduler, epoch, cfg, warmup=False):
    logger = logging.getLogger('global')

    model.cuda()
    model.train()
    world_size = 1
    rank = 0
    if args.dist:
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    def freeze_bn(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    model.apply(freeze_bn)
    logger.info('freeze bn')

    t0 = time.time()

    if args.dist:
        # update random seed
        train_loader.sampler.set_epoch(epoch)

    t0 = time.time()
    for iter, input in enumerate(train_loader):
        #torch.cuda.empty_cache()
        if warmup:
            # update lr for each iteration
            lr_scheduler.step()
        x = {
                'cfg': cfg,
                'image': torch.autograd.Variable(input[0]).cuda(),
                'image_info': input[1][:, :3],
                'ground_truth_bboxes': input[2],
                'ignore_regions': None, # input[3],
                'ground_truth_keypoints': input[4],
                'ground_truth_masks': input[5]
            }
        # for debug
        #debugger.store_tensor_as_image(input[0])
        #debugger.store_filenames(input[-1])
        t1 = time.time()

        outputs = model(x)
        t11 = time.time()

        rpn_cls_loss, rpn_loc_loss, rcnn_cls_loss, rcnn_loc_loss, keypoint_loss = outputs['losses']
        # gradient is averaged by normalizing the loss with world_size
        #loss = (rpn_cls_loss + rpn_loc_loss + rcnn_cls_loss + rcnn_loc_loss + keypoint_loss) / world_size
        loss = sum(outputs['losses']) / world_size

        '''
        if args.dist == 0 or dist.get_rank() == 0:
            graph = vis_helper.make_dot(loss, dict(model.named_parameters()))
            logger.info('PATH:{}'.format(os.environ['PATH']))
            graph.render(filename = 'graph', directory='graph', view=False)
        exit()
        '''
        t12 = time.time()
        lr_scheduler.optimizer.zero_grad()
        loss.backward()
        t13 = time.time()
        if args.dist:
            average_gradients(model)
        t14 = time.time()
        lr_scheduler.optimizer.step()
        t15 = time.time()

        rpn_accuracy = outputs['accuracy'][0][0] / 100.
        rcnn_accuracy = outputs['accuracy'][1][0] / 100.
        loss = loss.data.cpu()[0]
        rpn_cls_loss = rpn_cls_loss.data.cpu()[0]
        rpn_loc_loss = rpn_loc_loss.data.cpu()[0]
        rcnn_cls_loss = rcnn_cls_loss.data.cpu()[0]
        rcnn_loc_loss = rcnn_loc_loss.data.cpu()[0]
        if keypoint_loss is not None:
            keypoint_loss = keypoint_loss.data.cpu()[0]

        t2 = time.time()
        lr = lr_scheduler.get_lr()[0]
        logger.info('Epoch: [%d][%d/%d] LR:%f Time: %.3f Loss: %.5f (rpn_cls: %.5f rpn_loc: %.5f rpn_acc: %.5f'
                ' rcnn_cls: %.5f, rcnn_loc: %.5f rcnn_acc:%.5f kpt:%.5f)' %
              (epoch, iter, len(train_loader), lr, t2 - t0, loss * world_size,
               rpn_cls_loss, rpn_loc_loss, rpn_accuracy,
               rcnn_cls_loss, rcnn_loc_loss, rcnn_accuracy,
               keypoint_loss))
        t3 = time.time()
        #logger.info('data:{0}, forward:{1}, bp:{2}, sync:{3}, upd:{4}, loss:{5}, prt:{6}'.format(t1-t0, t11-t1, t13-t12, t14-t13, t15-t14, t2-t15, t3-t2))
        #logger.info('data:%f, ' % (t1-t0) +
        #            'forward:%f, ' % (t11-t1) +
        #            'sum_loss:%f, ' % (t12-t11) +
        #            'bp:%f, ' % (t13-t12) +
        #            'sync:%f, ' % (t14-t13) +
        #            'upd:%f, ' % (t15-t14) +
        #            'loss:%f, ' % (t2-t15) +
        #            'prt:%f, ' % (t3-t2))
        print_speed((epoch - 1) * len(train_loader) + iter + 1, t2 - t0, args.epochs * len(train_loader))
        t0 = t2

import datasets.pycocotools.mask as maskUtils

def write_results_to_file(writer, image_info, bboxes, keypoints, masks, mask_thresh=0.5, keep_num = 100):
    '''
        Args:
            writer: output stream to write results
            image_info:[B, 6] of list, for each image: (resized_img_h, resized_img_w, resize_scale, origin_image_h, origin_image_w, filename)
            bboxes: [N, 7] of np.array, each with (batch_index, x1, y1, x2, y2, score, cls)
            keypoints: [N, K, 3] of np.array, each keypoint holds (x, y, score)
            masks: [N] of list with elements of np.array
            masks is binaried by threshold 0.5 and converted to rle string to output
    '''
    logger = logging.getLogger('global')
    batch_size = len(image_info)
    for b_ix in range(batch_size):
        info = image_info[b_ix]
        img_resize_scale = info[2]
        #img_h = int(info[0]/img_resize_scale)
        #img_w = int(info[1]/img_resize_scale)
        img_h = int(info[3])
        img_w = int(info[4])
        img_id = info[5]
        keep_ix = np.where(bboxes[:, 0] == b_ix)[0]
        scores = bboxes[:, -2]
        keep_ix = sorted(keep_ix, key = lambda ix: scores[ix], reverse=True)
        keep_ix = keep_ix[:keep_num]
        for ix in keep_ix:
            res = {'image_id': int(img_id)}
            box = bboxes[ix]
            box[1:1+4] /= img_resize_scale
            box, box_score, cls = box[1:1+4], box[1+4], box[1+5]
            box = box.tolist()
            res['bbox'] = [box[0], box[1], box[2]-box[0],box[3]-box[1]]
            res['score'] = box_score.tolist()
            res['category_id'] = COCODataset.get_category(int(cls))

            if keypoints is not None:
                kpt = keypoints[ix]
                kpt[:, :2] /= img_resize_scale
                res['keypoints'] = kpt.reshape(-1).tolist()
            if masks is not None:
                msk = masks[ix]
                # msk = cv2.resize(msk, (img_w, img_h))
                msk = np.array(Image.fromarray(msk).resize((img_w, img_h)))
                mask = np.asfortranarray(msk > mask_thresh, dtype=np.uint8)
                rle = maskUtils.encode(mask)
                if isinstance(rle['counts'], bytes):
                    rle['counts'] = str(rle['counts'], encoding='utf-8')
                res['segmentation'] = rle
            writer.write(json.dumps(res) + '\n')
            writer.flush()
    return

def validate(val_loader, model, cfg):
    logger = logging.getLogger('global')
    try:
        rank = dist.get_rank()
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
    fout = open(os.path.join(args.results_dir, 'results.json.rank%d' % rank), 'w')

    for iter, input in enumerate(val_loader):
        img = torch.autograd.Variable(input[0]).cuda()
        img_info = input[1]
        gt_boxes = input[2]
        filenames = input[-1]
        x = {
                'cfg':cfg,
                'image': img,
                'image_info': img_info[:, :3],
                'ground_truth_bboxes': gt_boxes,
                'ignore_regions': None,
                'ground_truth_keypoints': None,
                'ground_truth_masks': None }
        batch_size = img.shape[0]
        t0 = time.time()
        outputs = model(x)['predict']
        t2 = time.time()


        proposals = outputs[0].data.cpu().numpy()
        bboxes = outputs[1].data.cpu().numpy()
        #keypoints = outputs[2].data.cpu().numpy()
        if isinstance(outputs[2], torch.autograd.Variable):
            keypoints = outputs[2].data.cpu().numpy()
            masks = None
        else:
            keypoints = None
            masks = outputs[2]
        # heatmap = outputs[3].data.cpu().numpy()
        if torch.is_tensor(gt_boxes):
            gt_boxes = gt_boxes.cpu().numpy()

        image_info = img_info
        img_ids = [_.split('/')[-1].split('_')[-1].split('.')[0] for _ in filenames]
        image_info = [list(x) + [y] for x, y in zip(image_info, img_ids)]

        # visualize results
        #vis_helper.vis_results(args.results_dir, image_info, bboxes, keypoints, masks, heatmap, cfg['shared']['class_names'])
        write_results_to_file(fout, image_info, bboxes, keypoints, masks, mask_thresh=0.5, keep_num=100)

        # rpn recall
        for b_ix in range(batch_size):
            rois_per_image = proposals[proposals[:, 0] == b_ix]
            gts_per_image = gt_boxes[b_ix]
            num_rc, num_gt = bbox_helper.compute_recall(rois_per_image[:,1:1+4], gts_per_image)
            total_gt += num_gt
            total_rc += num_rc
        logger.info('Test: [%d/%d] Time: %.3f %d/%d'%(iter, len(val_loader), t2-t0, total_rc, total_gt))
        print_speed(iter + 1, t2 - t0, len(val_loader))
    logger.info('rpn300 recall=%f'% (total_rc/total_gt))
    fout.close()
    return total_rc/total_gt

if __name__ == '__main__':
    main()
