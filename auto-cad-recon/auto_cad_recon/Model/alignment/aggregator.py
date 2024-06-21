#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn


class Aggregator(nn.Module):

    def __init__(self, shared_net=nn.Identity(), global_net=nn.Identity()):
        super().__init__()
        self.shared_net = shared_net
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                                         nn.Flatten(1))
        self.global_net = global_net

    def forward(self, features):
        features = self.shared_net(features)
        features = self.global_pool(features)
        return self.global_net(features)
