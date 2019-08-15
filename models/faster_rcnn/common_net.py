# -*- coding: utf-8 -*-
# @Time    : 18-4-19
# @Author  : Xinge
from .init import *
import torch
import torch.nn as nn
import cv2
import numpy as np


import torch.nn.init as init


# import numpy as np

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        # print m.__class__.__name__
        m.weight.data.normal_(0.0, 0.02)


def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.1)


class PixelNormLayer(nn.Module):
    def __init__(self, eps=1e-8):
        super(PixelNormLayer, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)

    def __repr__(self):
        return self.__class__.__name__ + '(eps = %s)' % (self.eps)


class Bias2d(nn.Module):
    def __init__(self, channels):
        super(Bias2d, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.bias.data.normal_(0, 0.002)

    def forward(self, x):
        n, c, h, w = x.size()
        return x + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(n, c, h, w)


##################################################################################
# Residual Blocks
##################################################################################
class INSResBlock(nn.Module):
    def conv3x3(self, inplanes, out_planes, stride=1):
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

    def __init__(self, inplanes, planes, stride=1, dropout=0.0):
        super(INSResBlock, self).__init__()
        model = []
        model += [self.conv3x3(inplanes, planes, stride)]
        model += [nn.InstanceNorm2d(planes)]
        model += [nn.ReLU(inplace=True)]
        model += [self.conv3x3(planes, planes)]
        model += [nn.InstanceNorm2d(planes)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class LinUnsRes(nn.Module):
    def __init__(self, channel=512, w=32, h=64):
        super(LinUnsRes, self).__init__()
        model = []
        model += [nn.Linear(4096, w * h, bias=None)]
        self.channel = channel
        self.w = w
        self.h = h
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):

        x1 = self.model(x)
        # x2 = torch.unsqueeze(x1, 0)
        try:
            out = x1.view(1, self.channel, self.w, self.h)
        except:
            print("x size: ", x.size())
            print("x1 size: ", x1.size())
            out = None
        return out


class LinUnsRes_cluster(nn.Module):
    def __init__(self, channel=128, w=64, h=64, cluster_num=4):
        super(LinUnsRes_cluster, self).__init__()
        # model = []
        # model += [nn.Conv1d(512, 512, kernel_size = 3, stride = 2, padding=1, bias=False)] # (4, 512, 4096) ==> (4, 512, 2048)
        self.channel = channel
        self.w = w
        self.h = h
        self.cluster_num = cluster_num
        # self.model = nn.Sequential(*model)
        # self.model.apply(gaussian_weights_init)

    def forward(self, x):

        # x1 = self.model(x)
        # x2 = torch.unsqueeze(x1, 0)
        try:
            out = x.view(self.cluster_num, self.channel, self.w, self.h)
        except:
            print("x size: ", x.size())
            # print("x1 size: ", x1.size())
            out = None
        return out


class LinUnsRes_cluster2(nn.Module):
    def __init__(self, channel=128, w=64, h=64, cluster_num=4):
        super(LinUnsRes_cluster2, self).__init__()
        model = []
        model += [nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1,
                            bias=False)]  # (4, 512, 4096) ==> (4, 512, 2048)
        self.channel = channel
        self.w = w
        self.h = h
        self.cluster_num = cluster_num
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):

        # x1 = self.model(x)
        # x2 = torch.unsqueeze(x1, 0)
        try:
            out = x.view(self.cluster_num, self.channel, self.w, self.h)
            out = self.model(out)  # (cluster_num, channel, w//2, h//2)

        except:
            print("x size: ", x.size())
            # print("x1 size: ", x1.size())
            out = None
        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)
        return x



class ResDis(nn.Module):
    def __init__(self, n_in=512, n_out=512, kernel_size=3, stride=2, padding=1, w=64, h=64):
        super(ResDis, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]  # 32
        model += [nn.LeakyReLU(inplace=True)]
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]  # 16
        model += [nn.LeakyReLU(inplace=True)]
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]  # 8
        # model += [nn.LeakyReLU(inplace=True)]
        self.w = w
        self.h = h

        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        # x = x.view()
        # x1 = self.model(x)
        x1 = torch.unsqueeze(x, 0)
        try:
            out1 = x1.view(1, 512, self.w, self.h)
            out2 = self.model(out1)
            out3 = nn.AvgPool2d(out2.size()[2:])(out2)
            out3 = torch.squeeze(out3)  # torch.size([512])
        except:
            print("x size: ", x.size())
            print("x1 size: ", x1.size())
            out3 = None
        return out3


class ResDis_cluster(nn.Module):
    def __init__(self, n_in=128, n_out=256, kernel_size=3, stride=2, padding=1, w=64, h=64, cluster_num=4):
        super(ResDis_cluster, self).__init__()
        self.w = w
        self.h = h
        self.cluster_num = cluster_num
        self.channel = n_in

        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]  # 32
        model += [nn.BatchNorm2d(num_features=n_out)]
        model += [nn.LeakyReLU(inplace=True)]

        n_in *= 2
        n_out *= 2

        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]  # 16
        model += [nn.BatchNorm2d(num_features=n_out)]
        model += [nn.LeakyReLU(inplace=True)]
        model += [nn.Conv2d(n_out, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]  # 8
        # model += [nn.LeakyReLU(inplace=True)]

        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x1):
        # x = x.view()
        # x.size(): (4, 128, 4096)
        # x1 = self.model(x)
        # x1 = torch.unsqueeze(x, 0)
        try:
            out1 = x1.view(self.cluster_num, self.channel, self.w, self.h)
            out2 = self.model(out1)
            out3 = nn.AvgPool2d(out2.size()[2:])(out2)  # size(4, 512)
            out3 = torch.squeeze(out3)  # torch.size([512])/
        except:
            # print("x size: ", x.size())
            print("x1 size: ", x1.size())
            # print("out3 size: ", out3.size())
            out3 = None
        return out3


##################################################################################
# Leaky ReLU-based conv layers
##################################################################################
class LeakyReLUConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(LeakyReLUConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class LeakyReLUConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, output_padding=0):
        super(LeakyReLUConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                     output_padding=output_padding, bias=False)]
        model += [nn.InstanceNorm2d(num_features=n_out)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class LeakyReLUConvTranspose2d_2(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, output_padding=0):
        super(LeakyReLUConvTranspose2d_2, self).__init__()
        model = []
        # model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
        model += [Interpolate(scale_factor=2, mode='bilinear')]
        model += [nn.Conv2d(in_channels=n_in, out_channels=n_out, kernel_size=kernel_size, padding=padding, stride=1,
                            bias=True)]
        model += [nn.InstanceNorm2d(num_features=n_out)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class LeakyReLUConvTranspose2d_4(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, output_padding=0):
        super(LeakyReLUConvTranspose2d_4, self).__init__()
        model = []
        # model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
        model += [nn.PixelShuffle(2)]
        model += [nn.Conv2d(in_channels=n_out, out_channels=n_out, kernel_size=kernel_size, padding=padding, stride=1,
                            bias=True)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class LeakyReLUBNConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(LeakyReLUBNConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        model += [nn.BatchNorm2d(n_out)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class LeakyReLUBNConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0, output_padding=0):
        super(LeakyReLUBNConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                     output_padding=output_padding, bias=False)]
        model += [nn.BatchNorm2d(n_out)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class LeakyReLUBNNSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(LeakyReLUBNNSConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        model += [nn.BatchNorm2d(n_out, affine=False)]
        model += [Bias2d(n_out)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class LeakyReLUBNNSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(LeakyReLUBNNSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        model += [nn.BatchNorm2d(n_out, affine=False)]
        model += [Bias2d(n_out)]
        model += [nn.LeakyReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


##################################################################################
# ReLU-based conv layers
##################################################################################
class ReLUINSConv2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(ReLUINSConv2d, self).__init__()
        model = []
        model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
        model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


class ReLUINSConvTranspose2d(nn.Module):
    def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
        super(ReLUINSConvTranspose2d, self).__init__()
        model = []
        model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                                     output_padding=output_padding, bias=True)]
        model += [nn.InstanceNorm2d(n_out, affine=False)]
        model += [nn.ReLU(inplace=True)]
        self.model = nn.Sequential(*model)
        self.model.apply(gaussian_weights_init)

    def forward(self, x):
        return self.model(x)


# class GaussianVAE2D(nn.Module):
#     def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
#         super(GaussianVAE2D, self).__init__()
#         self.en_mu = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
#         self.en_sigma = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
#         self.softplus = nn.Softplus()
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.en_mu.weight.data.normal_(0, 0.002)
#         self.en_mu.bias.data.normal_(0, 0.002)
#         self.en_sigma.weight.data.normal_(0, 0.002)
#         self.en_sigma.bias.data.normal_(0, 0.002)
#
#     def forward2(self, x):
#         # import pdb
#         # pdb.set_trace()
#         mu = self.en_mu(x)
#         sd = self.softplus(self.en_sigma(x))
#         return mu, mu, sd
#
#     def forward(self, x):
#         mu = self.en_mu(x)
#         sd = self.softplus(self.en_sigma(x))
#         if self.training:
#             noise = Variable(torch.randn(mu.size(0), mu.size(1), mu.size(2), mu.size(3))).cuda(x.data.get_device())
#             return mu + sd.mul(noise), mu, sd
#         else:
#             return mu, mu, sd


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# if __name__ == '__main__':
#     inp = Variable(torch.FloatTensor(1, 10, 224, 224))
#     ups1 = LeakyReLUConvTranspose2d(10, 3, 3, 2, 1, 1)
#     out1 = ups1(inp)
#     print (out1.size())
#
#     ups2 = LeakyReLUConvTranspose2d_2(10, 3, 3, 1, 1, 0)
#     out2 = ups2(inp)
#     print (out2.size())
