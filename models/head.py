import torch.nn as nn

class NaiveRpnHead(nn.Module):
    def __init__(self, inplanes, num_classes, num_anchors):
        '''
        Args:
            inplanes: input channel
            num_classes: as the name implies
            num_anchors: as the name implies
        '''
        super(NaiveRpnHead, self).__init__()
        self.num_anchors, self.num_classes = num_anchors, num_classes
        self.conv3x3 = nn.Conv2d(inplanes, 512, kernel_size=3, stride=1, padding=1)
        self.relu3x3 = nn.ReLU(inplace=True)
        self.conv_cls = nn.Conv2d(
            512, num_anchors * num_classes, kernel_size=1, stride=1)
        self.conv_loc = nn.Conv2d(
            512, num_anchors * 4, kernel_size=1, stride=1)

    def forward(self, x):
        '''
        Args:
            x: [B, inplanes, h, w], input feature
        Return:
            pred_cls: [B, num_anchors, h, w]
            pred_loc: [B, num_anchors*4, h, w]
        '''
        x = self.conv3x3(x)
        x = self.relu3x3(x)
        pred_cls = self.conv_cls(x)
        pred_loc = self.conv_loc(x)
        return pred_cls, pred_loc
