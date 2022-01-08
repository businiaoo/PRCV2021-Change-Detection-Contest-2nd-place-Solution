import torch.nn as nn

from models.block.Base import Conv3Relu


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = in_channels // 4
        self.head = nn.Sequential(Conv3Relu(in_channels, inter_channels),
                                  nn.Dropout(0.2),  # 使用0.1的dropout
                                  nn.Conv2d(inter_channels, out_channels, (1, 1)))

    def forward(self, x):

        return self.head(x)
