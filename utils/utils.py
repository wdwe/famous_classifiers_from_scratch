
class AverageMeter:
    """Computes and stores the average and current value
    taken from
    https://github.com/rwightman/pytorch-image-models/blob/ea2e59ca36380e90ee97c55280fd465ab98041af/timm/utils.py
    """
    def __init__(self):
        self.reset()

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