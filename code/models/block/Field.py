"""
    field  这个文件用来增大感受野
"""
import torch
from torch import nn
import torch.nn.functional as F

from models.block.Base import Conv3Relu


class PPM(nn.Module):
    def __init__(self, in_channels, sizes=(1, 2, 3, 6)):
        super(PPM, self).__init__()
        size1, size2, size3, size4 = sizes
        out_channels = int(in_channels / 4)
        self.pool1 = nn.Sequential(nn.AdaptiveAvgPool2d(size1),
                                   nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.pool2 = nn.Sequential(nn.AdaptiveAvgPool2d(size2),
                                   nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.pool3 = nn.Sequential(nn.AdaptiveAvgPool2d(size3),
                                   nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        self.pool4 = nn.Sequential(nn.AdaptiveAvgPool2d(size4),
                                   nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))

        self.dim_reduction = Conv3Relu(in_channels + out_channels * 4, in_channels)

    def forward(self, x):
        h, w = x.shape[-2:]

        feat1 = F.interpolate(self.pool1(x), (h, w), mode="bilinear", align_corners=True)
        feat2 = F.interpolate(self.pool2(x), (h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(self.pool3(x), (h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(self.pool4(x), (h, w), mode="bilinear", align_corners=True)

        out = self.dim_reduction(torch.cat((x, feat1, feat2, feat3, feat4), 1))
        return out


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates=(6, 12, 18)):
        super(ASPP, self).__init__()

        rate1, rate2, rate3 = tuple(atrous_rates)

        out_channels = int(in_channels / 2)

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate1, dilation=rate1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate2, dilation=rate2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=rate3, dilation=rate3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True))

        # 全局平均池化
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, (1, 1), bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

        self.dim_reduction = Conv3Relu(out_channels * 5, in_channels)

    def forward(self, x):
        h, w = x.shape[-2:]

        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)

        feat4 = F.interpolate(self.gap(x), (h, w), mode="bilinear", align_corners=True)

        out = self.dim_reduction(torch.cat((feat0, feat1, feat2, feat3, feat4), 1))

        return out


class SPP(nn.Module):
    def __init__(self, in_channels, sizes=(5, 9, 13)):
        super(SPP, self).__init__()
        size1, size2, size3 = sizes
        self.pool1 = nn.MaxPool2d(kernel_size=size1, stride=1, padding=size1//2)
        self.pool2 = nn.MaxPool2d(kernel_size=size2, stride=1, padding=size2//2)
        self.pool3 = nn.MaxPool2d(kernel_size=size3, stride=1, padding=size3//2)

        self.dim_reduction = Conv3Relu(in_channels * 4, in_channels)

    def forward(self, x):
        feat1 = self.pool1(x)
        feat2 = self.pool1(x)
        feat3 = self.pool1(x)

        out = self.dim_reduction(torch.cat([x, feat1, feat2, feat3], dim=1))
        return out
