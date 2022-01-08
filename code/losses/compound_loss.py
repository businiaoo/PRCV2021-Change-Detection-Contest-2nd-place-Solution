from losses.bce import *
from losses.dice import *


class BceDiceLoss(nn.Module):
    def __init__(self):
        super(BceDiceLoss, self).__init__()
        self.bce = BceLoss()
        self.dice = DiceLoss()

    def forward(self, logits, true):

        return self.bce(logits, true) + self.dice(logits, true)
