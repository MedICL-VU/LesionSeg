import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def dice_loss(input, target):
    # this loss function need input in the range (0, 1), and target in (0, 1)
    smooth = 0.01

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


def focal_loss(input, target, alpha, gamma, eps=1e-6):
    # this loss function need input in the range (0, 1), and target in (0, 1)
    input = input.view(-1, 1)
    input = torch.clamp(input, min=eps, max=1-eps)
    target = target.view(-1, 1)
    loss = -target * alpha * ((1 - input) ** gamma) * torch.log(input) - (1 - target) * (1-alpha) * (
                input ** gamma) * torch.log(1 - input)
    return loss.mean()


class FocalLoss(nn.Module):
    # this loss function need input in the range (-1, 1), and target in (0, 1)
    def __init__(self, gamma=0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        input = torch.unsqueeze(input, -1)
        input = torch.cat([0 - input, input], -1)
        input = input.contiguous().view(-1, 2)
        target = target.view(-1, 1).long()

        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss.mean()