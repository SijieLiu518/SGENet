"""
This code is refer from:
https://github.com/LBH1024/CAN/models/densenet.py

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(Bottleneck, self).__init__()
        interChannels = 4 * growthRate
        self.bn1 = nn.BatchNorm2d(interChannels)
        self.conv1 = nn.Conv2d(
            nChannels, interChannels, kernel_size=1,
            bias=True)  # Xavier initialization
        self.bn2 = nn.BatchNorm2d(growthRate)
        self.conv2 = nn.Conv2d(
            interChannels, growthRate, kernel_size=3, padding=1,
            bias=True)  # Xavier initialization
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        if self.use_dropout:
            out = self.dropout(out)
        out = torch.cat([x, out], 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate, use_dropout):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(
            nChannels, growthRate, kernel_size=3, padding=1, bias=False)

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = self.conv1(F.relu(x))
        if self.use_dropout:
            out = self.dropout(out)

        out = torch.cat([x, out], 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, out_channels, use_dropout):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(
            nChannels, out_channels, kernel_size=1, bias=False)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        if self.use_dropout:
            out = self.dropout(out)
        out = F.avg_pool2d(out, 2, ceil_mode=True, count_include_pad=False)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, reduction, bottleneck, use_dropout,
                 input_channel, **kwargs):
        super(DenseNet, self).__init__()

        nDenseBlocks = 16
        nChannels = 2 * growthRate

        self.conv1 = nn.Conv2d(
            input_channel,
            nChannels,
            kernel_size=7,
            padding=3,
            stride=2,
            bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        out_channels = int(math.floor(nChannels * reduction))
        self.trans1 = Transition(nChannels, out_channels, use_dropout)

        nChannels = out_channels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        nChannels += nDenseBlocks * growthRate
        out_channels = int(math.floor(nChannels * reduction))
        self.trans2 = Transition(nChannels, out_channels, use_dropout)

        nChannels = out_channels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks,
                                       bottleneck, use_dropout)
        self.out_channels = out_channels

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck,
                    use_dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, use_dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate, use_dropout))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, inputs):
        x, x_m, y = inputs
        out = self.conv1(x)
        out = F.relu(out, inplace=True)
        out = F.max_pool2d(out, 2, ceil_mode=True)
        out = self.dense1(out)
        out = self.trans1(out)
        out = self.dense2(out)
        out = self.trans2(out)
        out = self.dense3(out)
        return out, x_m, y
