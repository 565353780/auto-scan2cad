#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn


class ResNetBlock(nn.Module):

    def __init__(self,
                 num_channels,
                 kernel_size=3,
                 stride=1,
                 activation=nn.ReLU):
        super().__init__()

        padding = kernel_size // 2

        self.weight_block_0 = nn.Sequential(
            activation(inplace=True),
            nn.Conv3d(num_channels,
                      num_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding))
        self.weight_block_1 = nn.Sequential(
            activation(inplace=True),
            nn.Conv3d(num_channels,
                      num_channels,
                      kernel_size,
                      stride=stride,
                      padding=padding))
        return

    def forward(self, x):
        out = self.weight_block_0(x)
        out = self.weight_block_1(out)
        return x + out
