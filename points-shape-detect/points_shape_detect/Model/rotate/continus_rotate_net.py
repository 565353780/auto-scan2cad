#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn

from points_shape_detect.Method.rotate import (
    compute_geodesic_distance_from_two_matrices,
    compute_rotation_matrix_from_ortho6d)
from points_shape_detect.Method.weight import setWeight


class ContinusRotateNet(nn.Module):

    def __init__(self, infer=False):
        super().__init__()

        # bx#pointx3 -> bx1x1024
        self.feature_extracter = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1), nn.LeakyReLU(),
            nn.Conv1d(64, 128, kernel_size=1), nn.LeakyReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.AdaptiveMaxPool1d(output_size=1))

        # bx1024 -> bx6
        self.mlp = nn.Sequential(nn.Linear(1024, 512), nn.LeakyReLU(),
                                 nn.Linear(512, 6))

        self.mse_loss = nn.MSELoss()

        self.infer = infer
        return

    def encodeRotateMatrix(self, data):
        # B*N*3
        pt2 = data['inputs']['origin_query_point_array']

        B, N, _ = pt2.shape

        # Bx1024
        feature_pt2 = self.feature_extracter(pt2.transpose(1, 2)).view(B, -1)

        # Bx6
        rotation = self.mlp(feature_pt2)

        # Bx3x3
        rotate_matrix = compute_rotation_matrix_from_ortho6d(rotation)

        data['predictions']['rotation'] = rotation
        data['predictions']['rotate_matrix'] = rotate_matrix

        if not self.infer:
            data = self.lossRotate(data)

        if not self.infer:
            data = self.encodeCompleteRotateMatrix(data)
        return data

    def lossRotate(self, data):
        rotate_matrix = data['predictions']['rotate_matrix']
        gt_rotate_matrix = data['inputs']['rotate_matrix']

        loss_rotate_matrix = torch.pow(gt_rotate_matrix - rotate_matrix,
                                       2).mean()
        #  loss_geodesic = compute_geodesic_distance_from_two_matrices(
        #  gt_rotate_matrix, rotate_matrix).mean()

        data['losses']['loss_rotate_matrix'] = loss_rotate_matrix
        #  data['losses']['loss_geodesic'] = loss_geodesic
        return data

    def encodeCompleteRotateMatrix(self, data):
        # B*N*3
        pt2 = data['inputs']['origin_point_array']

        B, N, _ = pt2.shape

        # Bx1024
        feature_pt2 = self.feature_extracter(pt2.transpose(1, 2)).view(B, -1)

        # Bx6
        rotation = self.mlp(feature_pt2)

        # Bx3x3
        rotate_matrix = compute_rotation_matrix_from_ortho6d(rotation)

        data['predictions']['complete_rotation'] = rotation
        data['predictions']['complete_rotate_matrix'] = rotate_matrix

        if not self.infer:
            data = self.lossCompleteRotate(data)
        return data

    def lossCompleteRotate(self, data):
        rotate_matrix = data['predictions']['complete_rotate_matrix']
        gt_rotate_matrix = data['inputs']['rotate_matrix']

        loss_complete_rotate_matrix = torch.pow(
            gt_rotate_matrix - rotate_matrix, 2).mean()
        #  loss_complete_geodesic = compute_geodesic_distance_from_two_matrices(
        #  gt_rotate_matrix, rotate_matrix).mean()

        data['losses'][
            'loss_complete_rotate_matrix'] = loss_complete_rotate_matrix
        #  data['losses']['loss_complete_geodesic'] = loss_complete_geodesic
        return data

    @torch.no_grad()
    def rotateBackByPredict(self, data):
        pt2 = data['inputs']['origin_query_point_array']
        rotate_matrix = data['predictions']['rotate_matrix']

        rotate_matrix_inv = rotate_matrix.transpose(1, 2)

        B, N, _ = pt2.shape

        rotate_back_query_point_array = torch.bmm(pt2,
                                                  rotate_matrix_inv).detach()

        data['predictions'][
            'rotate_back_query_point_array'] = rotate_back_query_point_array
        return data

    def addWeight(self, data):
        if self.infer:
            return data

        data = setWeight(data, 'loss_rotate_matrix', 1)
        #  data = setWeight(data, 'loss_geodesic', 1)

        data = setWeight(data, 'loss_complete_rotate_matrix', 1)
        #  data = setWeight(data, 'loss_complete_geodesic', 1)
        return data

    def forward(self, data):
        data = self.encodeRotateMatrix(data)

        #  if self.infer:
        #  data = self.rotateBackByPredict(data)

        data = self.addWeight(data)
        return data
