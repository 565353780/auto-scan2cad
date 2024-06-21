#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import detectron2.layers as L


class SharedMLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size=None,
                 hidden_size=256,
                 num_hiddens=1,
                 activation=nn.ReLU,
                 output_activation=nn.Identity):
        super().__init__()

        assert num_hiddens > 0
        self.hiddens = nn.Sequential(*[
            L.Conv2d(in_channels=input_size if i == 0 else hidden_size,
                     out_channels=hidden_size,
                     kernel_size=(1, 1),
                     activation=activation(inplace=True))
            for i in range(num_hiddens)
        ])

        if output_size is not None:
            self.output = L.Conv2d(in_channels=hidden_size,
                                   out_channels=output_size,
                                   kernel_size=(1, 1))
            self.out_channels = output_size
        else:
            self.output = nn.Identity()
            self.out_channels = hidden_size

        self.output_activation = output_activation()

    def forward(self, x):
        return self.output_activation(self.output(self.hiddens(x)))
