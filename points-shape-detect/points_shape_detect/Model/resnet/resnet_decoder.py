#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from points_shape_detect.Model.resnet.resnet_block import ResNetBlock


class ResNetDecoder(nn.Module):

    def __init__(self, feats, relu_in=True):
        super().__init__()

        self.network = nn.Sequential(
            nn.ReLU() if relu_in else nn.Identity(),
            nn.ConvTranspose3d(feats[5], feats[4], 2),

            # 4 x 4 x 4
            ResNetBlock(feats[4]),
            nn.ConvTranspose3d(feats[4], feats[3], 4, stride=2, padding=1),

            # 8 x 8 x 8
            ResNetBlock(feats[3]),
            nn.ConvTranspose3d(feats[3], feats[2], 4, stride=2, padding=1),

            # 16 x 16 x 16
            ResNetBlock(feats[2]),
            nn.ConvTranspose3d(feats[2], feats[1], 4, stride=2, padding=1),

            # 32 x 32 x 32
            ResNetBlock(feats[1]),
            nn.ConvTranspose3d(feats[1], feats[1], 4, stride=2, padding=1),

            # 32 x 32 x 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(feats[1], feats[0], 7, stride=1, padding=3))
        return

    def forward(self, x):
        if x.ndim < 5:
            x = x.reshape(*x.size(), *(1 for _ in range(5 - x.ndim)))
        return self.network(x)
