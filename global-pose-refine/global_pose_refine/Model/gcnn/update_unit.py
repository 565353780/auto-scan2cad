#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

class UpdateUnit(nn.Module):

    def __init__(self, dim):
        super().__init__()

    def forward(self, target, source):
        assert target.size() == source.size(
        ), "source dimension must be equal to target dimension"
        update = target + source
        return update
