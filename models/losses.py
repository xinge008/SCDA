# -*- coding: utf-8 -*-
# @Time    : 18-4-19
# @Author  : Xinge
# import torch
import torch.nn as nn
# import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable, Function
# import numpy as np
from math import exp


class Losses(nn.Module):
    def __init__(self):
        super(Losses, self).__init__()
        # self.loss = nn.functional.kl_div


    def forward(self, input1, input2):
        """
        KL divergence loss
        :param input1:
        :param input2:
        :return:
        """
        # return 0.5 * (self.loss(input1, input2) + self.loss(input2, input1))
        # assert input1.size() == 2, "more than two dimensions"
        input1 = nn.functional.log_softmax(input1, dim = 1)
        input2 = nn.functional.softmax(input2, dim = 1)
        # loss_output = (input2 * (input2.log() - input1) ).sum() / input1.size(0)
        final_loss = (input2 * (input2.log() - input1.log())).mean()
        return final_loss * input1.size(0)

class Losses_triplet(nn.Module):
    def __init__(self):
        super(Losses_triplet, self).__init__()
        self.loss = nn.functional.kl_div


    def forward(self, real_img, input1, input2):
        """
        KL divergence loss
        :param input1: fake source
        :param input2: fake target
        :param real_img: real source
        :return:
        """
        # return 0.5 * (self.loss(input1, input2) + self.loss(input2, input1))
        # assert input1.size() == 2, "more than two dimensions"
        input1_log = nn.functional.log_softmax(input1, dim = 1)
        input2_log = nn.functional.log_softmax(input2, dim = 1)
        # input1 = nn.functional.softmax(input1, dim = 1)
        # input2 = nn.functional.softmax(input2, dim = 1)
        real_img = nn.functional.softmax(real_img, dim = 1)
        positive_loss = self.loss(input2_log, real_img, size_average=True) * 1000.0
        # negative_loss = torch.max(0, 1.0 - self.loss(input1_log, real_img, size_average=True))
        negative_loss = 1.0 - self.loss(input1_log, real_img, size_average=True) * 1000.0
        if (negative_loss.data < 0.0).all():
            negative_loss.data = torch.cuda.FloatTensor([0.0])
        # print("posi: ", positive_loss)
        # print("nega: ", negative_loss)
        # loss_output = (input2 * (input2.log() - input1) ).sum() / input1.size(0)
        return positive_loss + negative_loss

class Losses_triplet_nll(nn.Module):
    def __init__(self):
        super(Losses_triplet_nll, self).__init__()
        self.loss = nn.functional.mse_loss


    def forward(self, real_img, input1, input2):
        """
        KL divergence loss
        :param input1: fake source
        :param input2: fake target
        :param real_img: real source
        :return:
        """
        # return 0.5 * (self.loss(input1, input2) + self.loss(input2, input1))
        # assert input1.size() == 2, "more than two dimensions"
        posi_dist = self.loss(input2, real_img)
        nega_dist = self.loss(input1, real_img)

        Pt = torch.exp(nega_dist) / (torch.exp(nega_dist) + torch.exp(posi_dist))

        loss_pt = -1.0 * torch.log(Pt)

        return loss_pt


class GradReverse(Function):

    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x, lambd):
    return GradReverse(lambd)(x)


class Losses3(nn.Module):
    def __init__(self):
        super(Losses3, self).__init__()
        # self.loss = nn.functional.kl_div


    def forward(self, input1, input2):
        """
        KL divergence loss
        :param input1:
        :param input2:
        :return:
        """
        # return 0.5 * (self.loss(input1, input2) + self.loss(input2, input1))
        # assert input1.size() == 2, "more than two dimensions"
        input1 = nn.functional.log_softmax(input1, dim = 1)
        input2 = nn.functional.softmax(input2, dim = 1)
        loss_output = (input2 * (input2.log() - input1) ).sum() / input1.size(0)
        return loss_output

class Losses2(nn.Module):
    def __init__(self, in1_size, in2_size, out_size):
        super(Losses2, self).__init__()
        self.loss = nn.Bilinear(in1_size, in2_size, out_size, False)

    def forward(self, input1, input2):
        """
        Bilinear Transform Loss
        :param input1: (N, in1_size)
        :param input2: (N, in2_size)
        :return: (N, out_size)
        """
        return self.loss(input1, input2)




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=110, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=110, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

