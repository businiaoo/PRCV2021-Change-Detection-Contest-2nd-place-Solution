import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, avg=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])       # 对背景乘alpha，对前景乘(1-alpha)
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)   # 可以自己指定alpha
        self.avg = avg

    def forward(self, input, target):
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)

            # N,C,H*W => N,H*W,C
            input = input.transpose(1, 2)

            # N,H*W,C => N*H*W,C
            input = input.contiguous().view(-1, input.size(2))

        # (N,H,W) => (N*H*W,1)
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)   # (N*H*W,2)
        logpt = logpt.gather(1, target)   # (N*H*W,2) and (N*H*W,1)
        logpt = logpt.view(-1)  # (N*H*W,1) => (N*H*W)  (-inf,0]
        pt = Variable(logpt.data.exp())  # [0,1]

        if self.alpha is not None:            # is None
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt) ** self.gamma * logpt     # gamma==0

        # fore = loss[target.view(-1)==1].data
        # back = loss[target.view(-1)==0].data
        # print("fore: {} back: {} ".format(fore.sum(), back.sum()))   # 打印前景与背景的loss值

        if self.avg:
            return loss.mean()
        else:
            return loss


class BceLoss(FocalLoss):
    def __init__(self):
        super(BceLoss, self).__init__(gamma=0, alpha=None)   # 从FocalLoss继承过来

    def forward(self, input, target):
        loss = super(BceLoss, self).forward(input, target)
        return loss


class BceTopkLoss(FocalLoss):
    def __init__(self, k=10):  # 取前10%的损失
        super(BceTopkLoss, self).__init__(gamma=0, alpha=None, avg=False)   # average为False, 先不取平均，取完10%之后再平均
        self.k = k

    def forward(self, input, target):
        loss = super(BceTopkLoss, self).forward(input, target)

        # 这两行是topK的核心
        num_voxels = np.prod(loss.shape)
        loss, _ = torch.topk(loss.view(-1), int(num_voxels * self.k / 100), sorted=False)

        return loss.mean()


if __name__ == "__main__":
    pass
