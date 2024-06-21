#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from global_to_patch_retrieval.Model.resnet.encoder import ResNetEncoder
from global_to_patch_retrieval.Model.resnet.decoder import ResNetDecoder


class SeparationNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = ResNetEncoder(1, [16, 32, 64, 128, 512])
        self.decoder_fg = ResNetDecoder(1)
        self.decoder_bg = ResNetDecoder(1)
        return

    def encode(self, x):
        encoded = self.encoder(x)

        # Split latent space
        hidden_shape = list(encoded.size())
        hidden_shape[1] //= 2


        half_dim = self.encoder.num_features[-1] // 2
        split = torch.split(encoded, half_dim, dim=1)
        hidden_fg = split[0].reshape(hidden_shape)
        hidden_bg = split[1].reshape(hidden_shape)

        return hidden_fg, hidden_bg

    def forward(self, x):
        hidden_fg, hidden_bg = self.encode(x)

        decoded_fg = self.decoder_fg(hidden_fg)
        decoded_bg = self.decoder_bg(hidden_bg)
        return decoded_fg, decoded_bg
