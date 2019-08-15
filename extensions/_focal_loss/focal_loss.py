import torch
from torch.autograd import Function
from ._ext import focal_loss
import time
import logging

class SigmoidFocalLossFunction(Function):
    def __init__(self, gamma, alpha, num_classes):
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        
        self.weight_pos = None
        self.preds = None
        self.targets = None

    def forward(self, preds, targets, weight_pos):
        # preds shape: [Batch * h * w * num_anchors, num_classes]
        # targets shape: [Batch * h * w * num_anchors]
        preds_size  = preds.size()
        targets_size = targets.size()

        assert(preds_size[0] == targets_size[0])
        assert(preds_size[1] == self.num_classes)

        losses = preds.new(preds_size[0], preds_size[1]).zero_()
        weight_pos = float(weight_pos[0])
        N = preds_size[0] * preds_size[1]

        assert(losses.is_contiguous())
        assert(preds.is_contiguous())
        assert(targets.is_contiguous())

        assert(preds.is_cuda and targets.is_cuda)
        focal_loss.focal_loss_sigmoid_forward_cuda(N,
                                                   preds,
                                                   targets,
                                                   weight_pos,
                                                   self.gamma,
                                                   self.alpha,
                                                   self.num_classes,
                                                   losses)
        self.preds = preds
        self.targets = targets
        self.weight_pos = weight_pos
        return torch.cuda.FloatTensor([losses.sum()])

    def backward(self, grad_output):
        # grad_output: 1.0 / num_of_gpus
        preds_size = self.preds.size()
        grad_input = self.preds.new(preds_size[0], preds_size[1]).zero_()
        N = preds_size[0] * preds_size[1]

        assert(self.preds.is_contiguous())
        assert(self.targets.is_contiguous())
        assert(grad_input.is_contiguous())

        assert(self.preds.is_cuda and self.targets.is_cuda and grad_input.is_cuda)
        focal_loss.focal_loss_sigmoid_backward_cuda(N,
                                              self.preds,
                                              self.targets,
                                              grad_input,
                                              self.weight_pos,
                                              self.gamma,
                                              self.alpha,
                                              self.num_classes)
        grad_input = grad_input * grad_output
        return grad_input, None, None

class SoftmaxFocalLossFunction(Function):
    def __init__(self, gamma, alpha, num_classes):
        self.gamma = gamma
        self.alpha = alpha
        self.num_classes = num_classes
        
        self.weight_pos = None
        self.preds = None
        self.targets = None

    def forward(self, preds, targets, weight_pos):
        # preds shape: [Batch * h * w * num_anchors, num_classes]
        # targets shape: [Batch * h * w * num_anchors]
        preds_size  = preds.size()
        targets_size = targets.size()

        assert(preds_size[0] == targets_size[0])
        assert(preds_size[1] == self.num_classes)

        losses = preds.new(preds_size[0]).zero_()
        priors = preds.new(preds_size[0], preds_size[1]).zero_()

        weight_pos = float(weight_pos[0])
        N = preds_size[0] * preds_size[1]


        assert(losses.is_contiguous())
        assert(preds.is_contiguous())
        assert(targets.is_contiguous())
        assert(priors.is_contiguous())

        assert(preds.is_cuda and targets.is_cuda)
        focal_loss.focal_loss_softmax_forward_cuda(N,
                                                   preds,
                                                   targets,
                                                   weight_pos,
                                                   self.gamma,
                                                   self.alpha,
                                                   self.num_classes,
                                                   losses,
                                                   priors)

        self.preds = preds
        self.targets = targets
        self.weight_pos = weight_pos
        self.priors = priors
        return torch.cuda.FloatTensor([losses.sum()])

    def backward(self, grad_output):
        # grad_output: 1.0 / num_of_gpus
        preds_size = self.preds.size()
        grad_input = self.preds.new(preds_size[0], preds_size[1]).zero_()
        buff = self.preds.new(preds_size[0]).zero_()
        N = preds_size[0] * preds_size[1]

        assert(self.preds.is_contiguous())
        assert(self.targets.is_contiguous())
        assert(grad_input.is_contiguous())
        assert(buff.is_contiguous())

        assert(self.preds.is_cuda and self.targets.is_cuda and grad_input.is_cuda and buff.is_cuda)
        focal_loss.focal_loss_softmax_backward_cuda(N,
                                              self.preds,
                                              self.targets,
                                              grad_input,
                                              self.weight_pos,
                                              self.gamma,
                                              self.alpha,
                                              self.num_classes,
                                              self.priors,
                                              buff)
        grad_input = grad_input * grad_output
        return grad_input, None, None
