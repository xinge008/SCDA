# -*- coding: utf-8 -*-
# @Time    : 18-6-23 1:48
# @Author  : Xinge


import torch.nn as nn
import torch
# from torch.autograd import Variable
from common_net import *
import torch.nn.functional as F

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
    def __init__(self):
        super(GAN_dis_AE_patch, self).__init__()
        # for source domain only
        model_A_patch = [ResDis_cluster(n_in=128, n_out=256, kernel_size=3, stride=2, padding=1, w=64, h=64)]
        self.model_A_patch = nn.Sequential(*model_A_patch)
        # self.model_A_patch.apply(gaussian_weights_init)

    def forward(self, rois_features):
        out_C = self.model_A_patch(rois_features)
        out_C = torch.sigmoid(out_C) # size(4, 512)
        return out_C

# class GAN_dis_AE_patch_tar(nn.Module):
#     def __init__(self):
#         super(GAN_dis_AE_patch_tar, self).__init__()
#         # for source domain only
#         model_A_patch = [ResDis_cluster(n_in=512, n_out=512, kernel_size=3, stride=2, padding=1, w=64, h=64)]
#         self.model_A_patch = nn.Sequential(*model_A_patch)
#         self.model_A_patch.apply(gaussian_weights_init)
#
#     def forward(self, rois_features):
#         out_C = self.model_A_patch(rois_features)
#         out_C = torch.sigmoid(out_C)
#         return out_C

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

        tch = ch
        decB = []
        decA = []
        decB += [LinUnsRes_cluster(128, neww, newh)]
        decA += [LinUnsRes_cluster(128, neww, newh)]

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

        tch = ch
        decB = []
        decA = []
        decB += [LinUnsRes_cluster(128, neww, newh)]
        decA += [LinUnsRes_cluster(128, neww, newh)]

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


