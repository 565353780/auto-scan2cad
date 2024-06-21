#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from points_shape_detect.Model.encode.points_encoder import PointsEncoder
from points_shape_detect.Model.bbox.bbox_net import BBoxNet
from points_shape_detect.Model.complete.shape_complete_net import ShapeCompleteNet


class PointsShapeNet(nn.Module):

    def __init__(self, infer=False):
        super().__init__()
        self.points_encoder = PointsEncoder(infer)
        self.bbox_net = BBoxNet(infer)
        self.shape_complete_net = ShapeCompleteNet(infer)

        self.infer = infer
        return

    @torch.no_grad()
    def rotateBackByPredict(self, data):
        pt1 = data['inputs']['origin_point_array']
        pt2 = data['inputs']['origin_query_point_array']
        rotate_matrix = data['inputs']['rotate_matrix']

        rotate_matrix_inv = rotate_matrix.transpose(1, 2)

        B, N, _ = pt2.shape

        rotate_back_point_array = torch.bmm(pt1, rotate_matrix_inv).detach()

        rotate_back_query_point_array = torch.bmm(pt2,
                                                  rotate_matrix_inv).detach()

        data['inputs']['rotate_back_point_array'] = rotate_back_point_array
        data['inputs'][
            'rotate_back_query_point_array'] = rotate_back_query_point_array
        return data

    def forward(self, data):
        data = self.points_encoder(data)
        data = self.bbox_net(data)
        data = self.shape_complete_net(data)

        #  data = self.rotateBackByPredict(data)
        return data
