import torch.nn as nn

from losses.compound_loss import BceDiceLoss


class SelectLoss(nn.Module):
    def __init__(self, loss_name):
        super(SelectLoss, self).__init__()
        if loss_name == "bce+dice":
            self.loss = BceDiceLoss()
        else:
            raise Exception('Error. This loss function hasn\'t been defined: {}'.format(self.loss_name))

        # self.aux_loss = BceTopkLoss()    # 辅助分支的loss

    def forward(self, outs, labels, weight=(1, 1), aux_weight=0.4):
        loss = 0
        aux_loss = 0

        for i, label in enumerate(labels):
            for j, out in enumerate(outs[i::len(labels)]):
                if j == 0:
                    loss += self.loss(out, label) * weight[i]
                else:
                    aux_loss += self.loss(out, label) * weight[i]
        loss = loss / len(labels)   # 前后向loss求均值
        aux_loss = aux_loss / (len(outs) - len(labels)) * aux_weight if (len(outs) != len(labels)) else 0
        # print("loss: {} aux_loss: {}".format(loss, aux_loss))

        return loss + aux_loss


