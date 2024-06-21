#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models

resnets = {
    18: models.resnet18,
    34: models.resnet34,
    50: models.resnet50,
    101: models.resnet101,
    152: models.resnet152
}


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers=34, pretrained=True):
        super().__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        assert num_layers in resnets

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        return

    def forward(self, input_image):
        features = []
        #  x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(
            features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))
        return features
