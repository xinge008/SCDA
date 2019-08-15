from extensions import RoIPool
from .mask_rcnn import MaskRCNN
from models.head import NaiveRpnHead
from extensions._roi_align.modules.roi_align import RoIAlignAvg
import logging
logger = logging.getLogger('global')


import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from models.faster_rcnn.common_net import Interpolate


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

def init_weights(module, std = 0.01):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, std)


class ResNet(MaskRCNN):

    def __init__(self, block, layers, cfg):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # first stage: rpn
        num_anchors = len(cfg['anchor_scales']) * len(cfg['anchor_ratios'])
        self.rpn_head = NaiveRpnHead(1024, num_classes=2, num_anchors=num_anchors)

        # second stage: rcnn
        if cfg['roi_align']:
            self.roipooling = RoIAlignAvg(7, 7, 1.0/cfg['anchor_stride'])
        else:
            self.roipooling = RoIPool(7, 7, 1.0/cfg['anchor_stride'])

        # change stride to 1
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # add predict bbox layer
        self.fc_rcnn_cls = nn.Linear(512 * block.expansion, cfg['num_classes'])
        self.fc_rcnn_loc = nn.Linear(512 * block.expansion, cfg['num_classes'] * 4)

        # keypoint branch:
        if cfg['with_keypoint']:
            self.keypoint_roipooling = RoIAlignAvg(14, 14, 1.0/cfg['anchor_stride'])
            self.keypoint_head = self._make_branch(1024, 512, cfg['num_keypoints'], 8, upscaling=True)
        # mask branch 
        if cfg['with_mask']:
            self.mask_roipooling = RoIAlignAvg(14, 14, 1.0/cfg['anchor_stride'])
            self.mask_head = self._make_branch(1024, 256, cfg['num_classes'], 4, upscaling=False)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        init_weights(self.rpn_head)
        init_weights(self.fc_rcnn_cls, std=0.001)
        init_weights(self.fc_rcnn_loc, std=0.001)
        self.fix_layer_num=1
        self._fix_layer(self.fix_layer_num)

    def _make_branch(self, inplanes, midplanes, outplanes, depth, upscaling=False):
        ''' 
        Args:
            inplanes: input channel
            midplanes: intermediate channel
            outplanes: output channel
            depth: number of convs before upsample
        Return: a fcn branch
        '''
        planes = midplanes
        layers = []
        for i in range(depth):
            layers.append(nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)))
            inplanes = planes
        upsample_deconv = nn.Sequential(
            nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True))
        layers.append(upsample_deconv)
        if upscaling:
            # upsample_scale = nn.Upsample(scale_factor=2, mode='bilinear')
            upsample_scale = Interpolate(scale_factor=2, mode='bilinear')
            layers.append(upsample_scale)
        layers.append(nn.Conv2d(planes, outplanes, kernel_size=1))
        return nn.Sequential(*layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def train(self, mode=True):
        """Sets the module in training mode.

        This has any effect only on modules such as Dropout or BatchNorm.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.conv1.eval()
        self.bn1.eval()
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4][:self.fix_layer_num]:
            layer.eval()
        return self

    def _freeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = False

    def _fix_layer(self, layer_num):
        self._freeze_module(self.conv1)
        self._freeze_module(self.bn1)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4][:layer_num]:
            self._freeze_module(layer)

    def feature_extractor(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def rpn(self, x):
        rpn_pred_cls, rpn_pred_loc = self.rpn_head(x)
        return rpn_pred_cls, rpn_pred_loc

    def rcnn(self, x, rois):
        x = self.roipooling(x, rois)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pred_cls = self.fc_rcnn_cls(x)
        pred_loc = self.fc_rcnn_loc(x)
        return pred_cls, pred_loc

    def keypoint_predictor(self, x, rois): 
        assert(rois.shape[1] == 5)
        x = self.keypoint_roipooling(x, rois)
        x = self.keypoint_head(x)
        return x

    def mask_predictor(self, x, rois):
        assert(rois.shape[1] == 5)
        x = self.mask_roipooling(x, rois)
        x = self.mask_head(x)
        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
