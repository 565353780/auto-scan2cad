#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from global_to_patch_retrieval.Model.resnet.encoder import ResNetEncoder


class TripletNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_encoder = ResNetEncoder(1)
        return

    def forward(self, anchor, positive, negative):
        anchor = self.feature_encoder(anchor)
        positive = self.feature_encoder(positive)
        negative = self.feature_encoder(negative)
        return anchor, positive, negative

    def embed(self, data):
        return self.feature_encoder(data)
