#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import detectron2.layers as L


class MLP(nn.Module):

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
            L.Linear(in_features=input_size if i == 0 else hidden_size,
                     out_features=hidden_size) if i %
            2 == 0 else activation(inplace=True)
            for i in range(2 * num_hiddens)
        ])

        if output_size is not None:
            self.output = L.Linear(
                in_features=hidden_size,
                out_features=output_size,
            )
            self.out_features = output_size
        else:
            self.output = nn.Identity()
            self.out_features = hidden_size
        self.output_activation = output_activation()

    def forward(self, x):
        return self.output_activation(self.output(self.hiddens(x)))
