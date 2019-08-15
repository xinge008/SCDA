import torch
from torch.optim import Optimizer

class _IterLRScheduler(object):
    def __init__(self, optimizer, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_iter + 1)
        self.last_iter = last_iter

    def get_lr(self):
        raise NotImplementedError

    def step(self, iter=None):
        if iter is None:
            iter = self.last_iter + 1
        self.last_iter = iter
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class IterExponentialLR(_IterLRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma every iteration. When last_iter=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_iter (int): The index of last iter. Default: -1.
    """

    def __init__(self, optimizer, gamma, last_iter=-1):
        self.gamma = gamma
        super(IterExponentialLR, self).__init__(optimizer, last_iter)

    def get_lr(self):
        return [base_lr * self.gamma ** self.last_iter
                for base_lr in self.base_lrs]
