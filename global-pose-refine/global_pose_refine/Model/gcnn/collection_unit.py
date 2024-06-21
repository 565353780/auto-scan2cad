#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F


def normal_init(m, mean, stddev):
    m.weight.data.normal_(mean, stddev)
    m.bias.data.zero_()
    return True


class CollectionUnit(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        #  self.fc = nn.Linear(dim_in, dim_out, bias=True)
        #  normal_init(self.fc, 0, 0.01)

        self.fc = nn.Sequential(
            nn.Linear(dim_in, dim_in, bias=True),
            nn.ReLU(True),
            nn.Linear(dim_in, dim_out * 2, bias=True),
            nn.ReLU(True),
            nn.Linear(dim_out * 2, dim_out * 2, bias=True),
            nn.ReLU(True),
            nn.Linear(dim_out * 2, dim_out, bias=True),
            nn.ReLU(True),
            nn.Linear(dim_out, dim_out, bias=True),
        )
        return

    def forward(self, target, source, attention_base):
        #  assert attention_base.size(0) == source.size(
        #  0), "source number must be equal to attention number"
        fc_out = F.relu(self.fc(source))
        collect = torch.mm(attention_base, fc_out)  # Nobj x Nrel Nrel x dim
        collect_avg = collect / (
            attention_base.sum(1).view(collect.size(0), 1) + 1e-7)
        return collect_avg
