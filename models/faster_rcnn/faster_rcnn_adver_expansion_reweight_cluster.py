# -*- coding: utf-8 -*-
# @Time    : 18-6-22 9:31
# @Author  : Xinge

from functions.anchor_target import compute_anchor_targets
from functions.proposal_target import compute_proposal_targets
from functions.rpn_proposal import compute_rpn_proposals
from functions.predict_bbox import compute_predicted_bboxes
from functions.mask import compute_cluster_targets
import functools
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
from .common_net import *

logger = logging.getLogger('global')

class FasterRCNN_AdEx(nn.Module):
    def __init__(self, gan_model_flag):
        super(FasterRCNN_AdEx, self).__init__()


    def feature_extractor(self, x):
        raise NotImplementedError

    # def feature_extractor2(self, x):
    #     raise NotImplementedError

    def rpn(self, x):
        raise NotImplementedError

    def rcnn(self, x, rois):
        raise NotImplementedError

    def _add_rpn_loss(self, compute_anchor_targets_fn, rpn_pred_cls,
                      rpn_pred_loc):
        '''
        :param compute_anchor_targets_fn: functions to produce anchors' learning targets.
        :param rpn_pred_cls: [B, num_anchors * 2, h, w], output of rpn for classification.
        :param rpn_pred_loc: [B, num_anchors * 4, h, w], output of rpn for localization.
        :return: loss of classification and localization, respectively.
        '''
        # [B, num_anchors * 2, h, w], [B, num_anchors * 4, h, w]
        cls_targets, loc_targets, loc_masks, loc_normalizer = \
            compute_anchor_targets_fn(rpn_pred_loc.size())

        # tranpose to the input format of softmax_loss function
        rpn_pred_cls = rpn_pred_cls.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        cls_targets = cls_targets.permute(0, 2, 3, 1).contiguous().view(-1)
        rpn_loss_cls = F.cross_entropy(
            rpn_pred_cls, cls_targets, ignore_index=-1)
        # mask out negative anchors
        rpn_loss_loc = smooth_l1_loss_with_sigma(rpn_pred_loc * loc_masks,
                                                 loc_targets) / loc_normalizer

        # classification accuracy, top1
        acc = accuracy(rpn_pred_cls.data, cls_targets.data)[0]
        return rpn_loss_cls, rpn_loss_loc, acc

    def _add_rcnn_loss(self, rcnn_pred_cls, rcnn_pred_loc, cls_targets,
                       loc_targets, loc_weights):
        rcnn_loss_cls = F.cross_entropy(rcnn_pred_cls, cls_targets)
        loc_normalizer = cls_targets.shape[0]
        rcnn_loss_loc = smooth_l1_loss_with_sigma(rcnn_pred_loc * loc_weights,
                                                  loc_targets) / loc_normalizer
        acc = accuracy(rcnn_pred_cls, cls_targets)[0]
        return rcnn_loss_cls, rcnn_loss_loc, acc

    def _pin_args_to_fn(self, cfg, ground_truth_bboxes, image_info, ignore_regions):
        partial_fn = {}
        if self.training:
            partial_fn['anchor_target_fn'] = functools.partial(
                compute_anchor_targets,
                cfg=cfg['train_anchor_target_cfg'],
                ground_truth_bboxes=ground_truth_bboxes,
                ignore_regions=ignore_regions,
                image_info=image_info)
            partial_fn['proposal_target_fn'] = functools.partial(
                compute_proposal_targets,
                cfg=cfg['train_proposal_target_cfg'],
                ground_truth_bboxes=ground_truth_bboxes,
                ignore_regions=ignore_regions,
                image_info=image_info)
            partial_fn['rpn_proposal_fn'] = functools.partial(
                compute_rpn_proposals,
                cfg=cfg['train_rpn_proposal_cfg'],
                image_info=image_info)
        else:
            partial_fn['rpn_proposal_fn'] = functools.partial(
                compute_rpn_proposals,
                cfg=cfg['test_rpn_proposal_cfg'],
                image_info=image_info)
            partial_fn['predict_bbox_fn'] = functools.partial(
                compute_predicted_bboxes,
                image_info=image_info,
                cfg=cfg['test_predict_bbox_cfg'])
        return partial_fn

    def _compute_kl(self, mu, sd):
        mu_2 = torch.pow(mu, 2)
        sd_2 = torch.pow(sd, 2)
        encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / (mu_2.size(2) * mu_2.size(3))
        return encoding_loss

    def forward(self, input, target=None):
        '''
        Args:
            input: dict of input with keys of:
                'cfg': hyperparamters of faster-rcnn.
                'image': [b, 3, h, w], input data.
                'ground_truth_bboxes':[b, max_num_gts, 5] or None(self.training==False),
                                     each gt contains x1,y1,x2,y2,class.
                'image_info':[b, 3], resized_image_h, resized_image_w, resize_scale.
                'ignore_regions':[b,max_num_gts,4] or None.
        Return: dict of loss, predict, accuracy
        '''
        cfg = input['cfg']
        x_input = input['image']


        # x_input2.volatile = True
        # x_input2.requires_grad = False

        ground_truth_bboxes = input['ground_truth_bboxes']
        image_info = input['image_info']
        ignore_regions = input['ignore_regions']
        partial_fn = self._pin_args_to_fn(
            cfg,
            ground_truth_bboxes,
            image_info,
            ignore_regions)

        outputs = {'losses': [], 'predict': [], 'accuracy': []}
        x = self.feature_extractor(x_input) #torch.Size([1, 512, 32, 64])
        # x, mu, std = self.vae(x)
        # encoding_loss = 0.0
        # encoding_loss +=  self._compute_kl(mu, std) * 0.001
        # print ("encoding loss - 1: ", encoding_loss)
        # x_input2 =
        rpn_pred_cls, rpn_pred_loc = self.rpn(x)

        # rpn train function

        if self.training:
            # train rpn
            rpn_loss_cls, rpn_loss_loc, rpn_acc = \
                self._add_rpn_loss(partial_fn['anchor_target_fn'],
                                   rpn_pred_cls,
                                   rpn_pred_loc)
            # get rpn proposals
            compute_rpn_proposals_fn = partial_fn['rpn_proposal_fn']
            rpn_pred_cls = rpn_pred_cls.permute(0, 2, 3, 1).contiguous()
            rpn_pred_cls = F.softmax(rpn_pred_cls.view(-1, 2), dim=1).view_as(rpn_pred_cls)
            rpn_pred_cls = rpn_pred_cls.permute(0, 3, 1, 2)
            proposals = compute_rpn_proposals_fn(rpn_pred_cls.data, rpn_pred_loc.data)

            compute_proposal_target_fn = partial_fn['proposal_target_fn']
            rois, cls_targets, loc_targets, loc_weights = \
                compute_proposal_target_fn(proposals)
            assert (rois.shape[1] == 5)
            x_fea, rcnn_pred_cls, rcnn_pred_loc = self.rcnn(x, rois) # rois: (512, 5)
            # example rois: (924.3, 100.12, 930.8, 512.9), x1, y1, x2, y2
            x_cluster_fea, x_center_cluster = compute_cluster_targets(rois, x_fea, N_cluster=input['cluster_num'], threshold=input['threshold'])



            """
            RPN and rcnn for the target image
            """
            x_input2 = target
            x_gan = self.feature_extractor(x_input2)

            rpn_pred_cls_gan, rpn_pred_loc_gan = self.rpn(x_gan)
            # compute_rpn_proposals_fn = partial_fn['rpn_proposal_fn']
            rpn_pred_cls_gan = rpn_pred_cls_gan.permute(0, 2, 3, 1).contiguous()
            rpn_pred_cls_gan = F.softmax(rpn_pred_cls_gan.view(-1, 2), dim=1).view_as(rpn_pred_cls_gan)
            rpn_pred_cls_gan = rpn_pred_cls_gan.permute(0, 3, 1, 2)
            proposals_gan = compute_rpn_proposals_fn(rpn_pred_cls_gan.data, rpn_pred_loc_gan.data)

            # fast-rcnn test
            # assert ('proposal_target_fn' not in partial_fn)
            # predict_bboxes_fn = partial_fn['predict_bbox_fn']
            proposals_gan = proposals_gan[0:512, :5].cuda().contiguous()
            assert (proposals_gan.shape[1] == 5)
            x_fea_gan, rcnn_pred_cls_gan, rcnn_pred_loc_gan = self.rcnn(x_gan, proposals_gan)


            x_cluster_fea_gan, x_center_cluster_gan = compute_cluster_targets(proposals_gan, x_fea_gan, N_cluster=input['cluster_num'], threshold=input['threshold'])



            """
            additional part for domain adaptation, including domain transfer modules and distance loss
            """
            assert (x_gan.size() == x.size()), "gan_features does not match the backbone"

            rcnn_loss_cls, rcnn_loss_loc, rcnn_acc = self._add_rcnn_loss(
                rcnn_pred_cls, rcnn_pred_loc, cls_targets, loc_targets,
                loc_weights)

            outputs['losses'] = [rpn_loss_cls, rpn_loss_loc,
                                 rcnn_loss_cls, rcnn_loss_loc]
            outputs['accuracy'] = [rpn_acc, rcnn_acc]
            outputs['predict'] = [proposals]
            # outputs['features'] = [x, x_gan]
            if x_fea_gan.size(0) != 512:
                logger.info("Different channels {} at target image".format(x_fea_gan.size(0)))
                # outputs['ins_features'] = [x_fea, x_fea]
                outputs['cluster_features'] = [x_cluster_fea, x_cluster_fea]
                outputs['cluster_centers'] = [x_center_cluster, x_center_cluster]
            else:
                # outputs['ins_features'] = [x_fea, x_fea_gan]
                outputs['cluster_features'] = [x_cluster_fea, x_cluster_fea_gan]
                outputs['cluster_centers'] = [x_center_cluster, x_center_cluster_gan]

        else:
            # rpn test
            compute_rpn_proposals_fn = partial_fn['rpn_proposal_fn']
            rpn_pred_cls = rpn_pred_cls.permute(0, 2, 3, 1).contiguous()
            rpn_pred_cls = F.softmax(rpn_pred_cls.view(-1, 2), dim=1).view_as(rpn_pred_cls)
            rpn_pred_cls = rpn_pred_cls.permute(0, 3, 1, 2)
            proposals = compute_rpn_proposals_fn(rpn_pred_cls.data, rpn_pred_loc.data)

            # fast-rcnn test
            assert ('proposal_target_fn' not in partial_fn)
            predict_bboxes_fn = partial_fn['predict_bbox_fn']
            proposals = proposals[:, :5].cuda().contiguous()
            assert (proposals.shape[1] == 5)
            x_fea, rcnn_pred_cls, rcnn_pred_loc = self.rcnn(x, proposals)
            rcnn_pred_cls = F.softmax(rcnn_pred_cls, dim=1)
            bboxes = predict_bboxes_fn(proposals, rcnn_pred_cls,
                                       rcnn_pred_loc)
            outputs['predict'] = [proposals, bboxes]
        return outputs


def smooth_l1_loss_with_sigma(pred, targets, sigma=3.0):
    sigma_2 = sigma**2
    diff = pred - targets
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    loss = torch.pow(diff, 2) * sigma_2 / 2. * smoothL1_sign \
            + (abs_diff - 0.5 / sigma_2) * (1. - smoothL1_sign)
    reduced_loss = torch.sum(loss)
    return reduced_loss


def accuracy(output, target, topk=(1, ), ignore_index=-1):
    """Computes the precision@k for the specified values of k"""
    keep = torch.nonzero(target != ignore_index).squeeze()
    #logger.info('target.shape:{0}, keep.shape:{1}'.format(target.shape, keep.shape))
    assert (keep.dim() == 1)
    target = target[keep]
    output = output[keep]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class GAN_dis_AE(nn.Module):
    def __init__(self, params):
        super(GAN_dis_AE, self).__init__()
        ch = params['ch']  # 32
        input_dim_a = params['input_dim_a']  # 3

        n_layer = params['n_layer'] # 5
        self.model_A = self._make_net(ch, input_dim_a, n_layer - 1)  # for the first stage
        self.model_A.apply(gaussian_weights_init)
        self.model_B = self._make_net(ch, input_dim_a, n_layer - 1)  # for the first stage
        self.model_B.apply(gaussian_weights_init)



    def _make_net(self, ch, input_dim, n_layer):
        model = []
        model += [LeakyReLUConv2d(input_dim, ch, kernel_size=3, stride=2, padding=1)]  # 16
        tch = ch
        for i in range(0, n_layer):
            model += [LeakyReLUConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]  # 8
            tch *= 2
        model += [nn.Conv2d(tch, 1, kernel_size=1, stride=1, padding=0)]  # 1
        return nn.Sequential(*model)

    def forward(self, x_aa, x_bb):
        """
        :param x_bA: the concatenation of
        :param x_aB:
        :param rois_feature: (512 x 4096)
        :return:
        """
        # x_aa, x_bb = torch.split(x_A, x_A.size(0) // 2, 0)
        out_A = self.model_A(x_aa)
        out_A = out_A.view(out_A.size(0), -1)
        out_B = self.model_B(x_bb)
        out_B = out_B.view(out_B.size(0), -1)

        # out = torch.cat((out_A, out_B), 0)
        return out_A, out_B



class GAN_dis_AE_patch(nn.Module):
    def __init__(self, params = None):
        super(GAN_dis_AE_patch, self).__init__()
        # for source domain only
        if params:
            self.n_in = params['n_in']
            self.n_out = params['n_out']
            cluster_num1 = params['cluster_num']
        else:
            self.n_in = 128
            self.n_out = 256
            cluster_num1 = 4


        model_A_patch = [ResDis_cluster(n_in=self.n_in, n_out=self.n_out, kernel_size=3, stride=2, padding=1, w=64, h=64, cluster_num=cluster_num1)]
        self.model_A_patch = nn.Sequential(*model_A_patch)
        # self.model_A_patch.apply(gaussian_weights_init)

    def forward(self, rois_features):
        out_C = self.model_A_patch(rois_features)
        out_C = torch.sigmoid(out_C) # size(4, 512)
        return out_C


class GAN_decoder_AE(nn.Module):
    def __init__(self, params):
        super(GAN_decoder_AE, self).__init__()
        input_dim_b = params['input_dim_b']
        ch = params['ch'] # 32
        # n_gen_shared_blk = params['n_gen_shared_blk']
        n_gen_res_blk    = params['n_gen_res_blk']   # 4
        n_gen_front_blk  = params['n_gen_front_blk'] # 3
        if 'res_dropout_ratio' in params.keys():
            res_dropout_ratio = params['res_dropout_ratio']
        else:
            res_dropout_ratio = 0

        # self.embedding1= nn.Linear(4096, 2048, bias=None)
        # self.embedding2 = nn.Linear(4096, 2048, bias=None)
        if 'neww' in params.keys():
            neww = params['neww']
        else:
            neww = 64

        if 'newh' in params.keys():
            newh = params['newh']
        else:
            newh = 64

        if 'cluster_num' in params.keys():
            cluster_num = params['cluster_num']
        else:
            cluster_num = 4

        tch = ch
        decB = []
        decA = []
        decB += [LinUnsRes_cluster(ch, neww, newh, cluster_num)]
        decA += [LinUnsRes_cluster(ch, neww, newh, cluster_num)]

        for i in range(0, n_gen_res_blk):
            decB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
            decA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
        for i in range(0, n_gen_front_blk-1):
            decB += [LeakyReLUConvTranspose2d_2(tch, tch//2, kernel_size=3, stride=1, padding=1, output_padding=0)]
            decA += [LeakyReLUConvTranspose2d_2(tch, tch//2, kernel_size=3, stride=1, padding=1, output_padding=0)]
            tch = tch//2
        # decB += [nn.Conv2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
        decB += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
        decA += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
        decB += [nn.Tanh()]
        decA += [nn.Tanh()]

        # decB += [nn.LeakyReLU(inplace=True)]
        # self.dec_shared = nn.Sequential(*dec_shared)
        self.decode_B = nn.Sequential(*decB)
        self.decode_B.apply(gaussian_weights_init)
        self.decode_A = nn.Sequential(*decA)
        self.decode_A.apply(gaussian_weights_init)

    def forward(self, x_aa, x_bb):
        # x_aa and x_bb is 512 x 4096 ==> 512 x 64 x 64
        # out = self.dec_shared(x_A)
        # x_aa, x_bb = torch.split(x_A, x_A.size(0) // 2, 0)
        out1 = self.decode_A(x_aa)
        out2 = self.decode_B(x_bb)
        # out = torch.cat((out1, out2), 0)
        return out1, out2


class GAN_decoder_AE_32(nn.Module):
    def __init__(self, params):
        super(GAN_decoder_AE_32, self).__init__()
        input_dim_b = params['input_dim_b']
        ch = params['ch'] # 32
        # n_gen_shared_blk = params['n_gen_shared_blk']
        n_gen_res_blk    = params['n_gen_res_blk']   # 4
        n_gen_front_blk  = params['n_gen_front_blk'] # 3
        if 'res_dropout_ratio' in params.keys():
            res_dropout_ratio = params['res_dropout_ratio']
        else:
            res_dropout_ratio = 0

        # self.embedding1= nn.Linear(4096, 2048, bias=None)
        # self.embedding2 = nn.Linear(4096, 2048, bias=None)
        if 'neww' in params.keys():
            neww = params['neww']
        else:
            neww = 64

        if 'newh' in params.keys():
            newh = params['newh']
        else:
            newh = 64

        if 'cluster_num' in params.keys():
            cluster_num = params['cluster_num']
        else:
            cluster_num = 4

        tch = ch
        decB = []
        decA = []
        decB += [LinUnsRes_cluster2(ch, neww, newh, cluster_num)]
        decA += [LinUnsRes_cluster2(ch, neww, newh, cluster_num)]

        for i in range(0, n_gen_res_blk):
            decB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
            decA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
        for i in range(0, n_gen_front_blk-1):
            decB += [LeakyReLUConvTranspose2d_2(tch, tch//2, kernel_size=3, stride=1, padding=1, output_padding=0)]
            decA += [LeakyReLUConvTranspose2d_2(tch, tch//2, kernel_size=3, stride=1, padding=1, output_padding=0)]
            tch = tch//2
        # decB += [nn.Conv2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
        decB += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
        decA += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
        decB += [nn.Tanh()]
        decA += [nn.Tanh()]

        # decB += [nn.LeakyReLU(inplace=True)]
        # self.dec_shared = nn.Sequential(*dec_shared)
        self.decode_B = nn.Sequential(*decB)
        self.decode_B.apply(gaussian_weights_init)
        self.decode_A = nn.Sequential(*decA)
        self.decode_A.apply(gaussian_weights_init)

    def forward(self, x_aa, x_bb):
        # x_aa and x_bb is 512 x 4096 ==> 512 x 64 x 64
        # out = self.dec_shared(x_A)
        # x_aa, x_bb = torch.split(x_A, x_A.size(0) // 2, 0)
        out1 = self.decode_A(x_aa)
        out2 = self.decode_B(x_bb)
        # out = torch.cat((out1, out2), 0)
        return out1, out2


class GAN_decoder_AE_de(nn.Module):
    def __init__(self, params):
        super(GAN_decoder_AE_de, self).__init__()
        input_dim_b = params['input_dim_b']
        ch = params['ch']  # 32
        # n_gen_shared_blk = params['n_gen_shared_blk']
        n_gen_res_blk = params['n_gen_res_blk']  # 3
        n_gen_front_blk = params['n_gen_front_blk']  # 4
        if 'res_dropout_ratio' in params.keys():
            res_dropout_ratio = params['res_dropout_ratio']
        else:
            res_dropout_ratio = 0

        # self.embedding1= nn.Linear(4096, 2048, bias=None)
        # self.embedding2 = nn.Linear(4096, 2048, bias=None)
        if 'neww' in params.keys():
            neww = params['neww']
        else:
            neww = 64

        if 'newh' in params.keys():
            newh = params['newh']
        else:
            newh = 64

        if 'cluster_num' in params.keys():
            cluster_num = params['cluster_num']
        else:
            cluster_num = 4


        tch = ch
        decB = []
        decA = []
        decB += [LinUnsRes_cluster2(ch, neww, newh, cluster_num)]
        decA += [LinUnsRes_cluster2(ch, neww, newh, cluster_num)]

        for i in range(0, n_gen_res_blk):
            decB += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
            decA += [INSResBlock(tch, tch, dropout=res_dropout_ratio)]
        for i in range(0, n_gen_front_blk - 1):
            decB += [LeakyReLUConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
            decA += [LeakyReLUConvTranspose2d(tch, tch // 2, kernel_size=3, stride=2, padding=1, output_padding=1)]
            tch = tch // 2
        # decB += [nn.Conv2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
        decB += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
        decA += [nn.ConvTranspose2d(tch, input_dim_b, kernel_size=1, stride=1, padding=0)]
        decB += [nn.Tanh()]
        decA += [nn.Tanh()]

      # decB += [nn.LeakyReLU(inplace=True)]
      # self.dec_shared = nn.Sequential(*dec_shared)
        self.decode_B = nn.Sequential(*decB)
        self.decode_B.apply(gaussian_weights_init)
        self.decode_A = nn.Sequential(*decA)
        self.decode_A.apply(gaussian_weights_init)


    def forward(self, x_aa, x_bb):
        # x_aa and x_bb is 512 x 4096 ==> 512 x 64 x 64
        # out = self.dec_shared(x_A)
        # x_aa, x_bb = torch.split(x_A, x_A.size(0) // 2, 0)
        out1 = self.decode_A(x_aa)
        out2 = self.decode_B(x_bb)
        # out = torch.cat((out1, out2), 0)
        return out1, out2


