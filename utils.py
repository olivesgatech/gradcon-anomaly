import os

import shutil
import torch


class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.prev = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, checkpointdir):
    fullpath = os.path.join(checkpointdir, 'checkpoint.pth.tar')
    fullpath_best = os.path.join(checkpointdir, 'model_best.pth.tar')
    torch.save(state, fullpath)

    if is_best:
        shutil.copyfile(fullpath, fullpath_best)
