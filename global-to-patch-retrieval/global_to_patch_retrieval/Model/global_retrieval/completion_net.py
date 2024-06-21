#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from global_to_patch_retrieval.Model.resnet.encoder import ResNetEncoder
from global_to_patch_retrieval.Model.resnet.decoder import ResNetDecoder


class CompletionNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = ResNetEncoder(1)
        self.decoder = ResNetDecoder(1)
        return

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return out
