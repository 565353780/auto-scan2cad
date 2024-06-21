#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from global_to_patch_retrieval.Model.resnet.block import ResNetBlock


class ResNetEncoder(nn.Module):

    def __init__(self, num_input_channels=1, num_features=None):
        super().__init__()

        if num_features is None:
            num_features = [8, 16, 32, 64, 256]

        self.num_features = [num_input_channels] + num_features

        self.network = nn.Sequential(
            # 32 x 32 x 32
            nn.Conv3d(self.num_features[0],
                      self.num_features[1],
                      kernel_size=7,
                      stride=1,
                      padding=3,
                      bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),

            # 32 x 32 x 32
            nn.Conv3d(self.num_features[1],
                      self.num_features[1],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            ResNetBlock(self.num_features[1]),

            # 16 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),
            nn.Conv3d(self.num_features[1],
                      self.num_features[2],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            ResNetBlock(self.num_features[2]),

            # 8 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[2]),
            nn.Conv3d(self.num_features[2],
                      self.num_features[3],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            ResNetBlock(self.num_features[3]),

            # 4 x 4 x 4
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[3]),
            nn.Conv3d(self.num_features[3],
                      self.num_features[4],
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            ResNetBlock(self.num_features[4]),

            # 2 x 2 x 2
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[4]),
            nn.Conv3d(self.num_features[4],
                      self.num_features[5],
                      kernel_size=2,
                      stride=1,
                      padding=0,
                      bias=False),
            # FIXME: here can not add a batch norm in network!
            #  nn.BatchNorm3d(self.num_features[5]),
        )

        self.init_weights()
        return

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feature = self.network(x)
        return feature
