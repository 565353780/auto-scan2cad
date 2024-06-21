#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scene_layout_detect.Model.encoder.resnet_encoder import ResnetEncoder
from scene_layout_detect.Model.decoder.depth_decoder import DepthDecoder


class LayoutNet(nn.Module):

    def __init__(self):
        self.encoder = ResNetEncoder()
        self.decoder = DepthDecoder()
        return

    def forward(self, data):
        return data
