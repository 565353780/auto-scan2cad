#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from image_to_cad.Model.retrieval.retrieval_network import RetrievalNetwork
from image_to_cad.Model.retrieval.resnet_block import ResNetBlock


class ResNetEncoder(RetrievalNetwork):

    def __init__(self, relu_out=False, embedding_dim=256):
        super().__init__()

        self.feats = feats = [1, 8, 16, 32, 64, embedding_dim]

        self.network = nn.Sequential(
            # 32 x 32 x 32
            nn.Conv3d(feats[0], feats[1], 7, stride=1, padding=3),
            nn.ReLU(inplace=True),

            # 32 x 32 x 32
            nn.Conv3d(feats[1], feats[1], 4, stride=2, padding=1),
            ResNetBlock(feats[1]),

            # 16 x 16 x 16
            nn.ReLU(inplace=True),
            nn.Conv3d(feats[1], feats[2], 4, stride=2, padding=1),
            ResNetBlock(feats[2]),

            # 8 x 8 x 8
            nn.ReLU(inplace=True),
            nn.Conv3d(feats[2], feats[3], 4, stride=2, padding=1),
            ResNetBlock(feats[3]),

            # 4 x 4 x 4
            nn.ReLU(inplace=True),
            nn.Conv3d(feats[3], feats[4], 4, stride=2, padding=1),
            ResNetBlock(feats[4]),

            # 2 x 2 x 2
            nn.ReLU(inplace=True),
            nn.Conv3d(feats[4], feats[5], 2, stride=1),

            # Flatten to a vector
            nn.Flatten(1),

            # Relu out or not
            nn.ReLU(inplace=True) if relu_out else nn.Identity())
        return

    def forward(self, x):
        return self.network(x)

    @property
    def embedding_dim(self):
        return self.feats[-1]
