#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from auto_cad_recon.Model.alignment.aggregator import Aggregator
from auto_cad_recon.Model.alignment.mlp import MLP
from auto_cad_recon.Model.alignment.shared_mlp import SharedMLP

from auto_cad_recon.Loss.loss_functions import l1_loss, l2_loss


class AlignmentHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")

        self.num_classes = 9
        self.output_grid_size = 32

        self.input_size = 32
        shape_code_size = 512

        encode_shared_net = SharedMLP(self.input_size,
                                      hidden_size=shape_code_size,
                                      num_hiddens=2)

        encode_global_net = MLP(encode_shared_net.out_channels,
                                hidden_size=shape_code_size,
                                num_hiddens=2)

        self.shape_encoder = Aggregator(encode_shared_net, encode_global_net)
        self.shape_code_drop = nn.Dropout(0.3)

        self.scale_head = MLP(shape_code_size, 3, num_hiddens=2)

        self.trans_head = MLP(shape_code_size, 3, num_hiddens=1)

        self.noc_head = MLP(shape_code_size,
                            pow(self.input_size, 3),
                            hidden_size=shape_code_size,
                            num_hiddens=4)

        self.rot_head = MLP(shape_code_size, 4, num_hiddens=2)

        self.min_nocs = 16
        return

    def encode_shape(self, data):
        # shape_features 32,32,32
        # shape_code 512
        shape_code = self.shape_encoder(data['predictions']['point_udf'])

        data['predictions']['shape_code'] = self.shape_code_drop(shape_code)
        return data

    def forward_scale(self, data):
        data['predictions']['delta_scale'] = self.scale_head(
            data['predictions']['shape_code'])

        data['predictions']['scale'] = torch.mul(
            data['predictions']['init_scale'],
            data['predictions']['delta_scale'])

        if self.training:
            data = self.scale_loss(data)
        return data

    def scale_loss(self, data):
        delta_scale_gt = torch.div(data['inputs']['scale_inv'],
                                   data['predictions']['init_scale'])

        data['losses']['loss_scale'] = l1_loss(
            data['predictions']['delta_scale'], delta_scale_gt)
        return data

    def forward_trans(self, data):
        data['predictions']['delta_translate'] = self.trans_head(
            data['predictions']['shape_code'])

        data['predictions']['translate'] = torch.add(
            data['predictions']['init_translate'],
            data['predictions']['delta_translate'],
            alpha=1)

        if self.training:
            data = self.trans_loss(data)
        return data

    def trans_loss(self, data):
        delta_translate_gt = torch.sub(data['inputs']['translate_inv'],
                                       data['predictions']['init_translate'],
                                       alpha=1)

        data['losses']['loss_translate'] = l2_loss(
            data['predictions']['delta_translate'], delta_translate_gt)
        return data

    def forward_noc(self, data):
        data['predictions']['cad_udf'] = self.noc_head(
            data['predictions']['shape_code']).view(-1, self.input_size,
                                                    self.input_size,
                                                    self.input_size)

        if self.training:
            data = self.noc_loss(data)
        return data

    def noc_loss(self, data):
        data['losses']['loss_noc'] = l1_loss(data['predictions']['cad_udf'],
                                             data['inputs']['cad_udf'])
        return data

    def forward_proc(self, data):
        data['predictions']['rotate'] = self.rot_head(
            data['predictions']['shape_code'])

        if self.training:
            data = self.proc_loss(data)
        return data

    def proc_loss(self, data):
        data['losses']['loss_proc'] = l1_loss(data['predictions']['rotate'],
                                              data['inputs']['rotate_inv'])
        return data

    def forward(self, data):
        data = self.encode_shape(data)

        data = self.forward_scale(data)

        data = self.forward_trans(data)

        data = self.forward_noc(data)

        data = self.forward_proc(data)
        return data
