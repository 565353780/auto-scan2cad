#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import nn

from points_shape_detect.Model.encode.pc_transformer import PCTransformer

from points_shape_detect.Method.rotate import (
    compute_geodesic_distance_from_two_matrices,
    compute_rotation_matrix_from_ortho6d)
from points_shape_detect.Method.weight import setWeight


class TransformerRotateNet(nn.Module):

    def __init__(self, infer=False):
        super().__init__()
        #M
        self.num_query = 96
        #C
        self.trans_dim = 384
        self.knn_layer = 1

        self.feature_encoder = PCTransformer(in_chans=3,
                                             embed_dim=self.trans_dim,
                                             depth=[6, 8],
                                             drop_rate=0.,
                                             num_query=self.num_query,
                                             knn_layer=self.knn_layer)

        self.increase_dim = nn.Sequential(nn.Conv1d(self.trans_dim, 1024, 1),
                                          nn.BatchNorm1d(1024),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          nn.Conv1d(1024, 1024, 1))

        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)

        self.rotate_feature_encoder = nn.Sequential(
            nn.Conv1d(self.num_query * self.trans_dim, self.trans_dim, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1))

        self.rotate_decoder = nn.Sequential(nn.Conv1d(self.trans_dim, 6, 1),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv1d(6, 6, 1))

        self.l1_loss = nn.SmoothL1Loss()

        self.infer = infer
        return

    def encodeShape(self, data):
        # Bx#pointx3
        origin_query_point_array = data['inputs']['origin_query_point_array']

        # Bx#pointx3 -[feature_encoder]-> BxMxC and BxMx3
        origin_encode_feature, origin_coarse_point_cloud = self.feature_encoder(
            origin_query_point_array)

        B, M, C = origin_encode_feature.shape

        # BxMxC -[transpose]-> BxCxM -[increase_dim]-> Bx1024xM -[transpose]-> BxMx1024
        origin_points_feature = self.increase_dim(
            origin_encode_feature.transpose(1, 2)).transpose(1, 2)

        # BxMx1024 -[max]-> Bx1024
        origin_global_points_feature = torch.max(origin_points_feature,
                                                 dim=1)[0]

        # Bx1024 -[unsqueeze]-> Bx1x1024 -[expand]-> BxMx1024
        origin_replicate_global_points_feature = origin_global_points_feature.unsqueeze(
            -2).expand(-1, M, -1)

        # BxMx1024 + BxMxC + BxMx3 -[cat]-> BxMx(C+1027)
        origin_global_feature = torch.cat([
            origin_replicate_global_points_feature, origin_encode_feature,
            origin_coarse_point_cloud
        ],
                                          dim=-1)

        # BxMx(C+1027) -[reshape]-> BMx(C+1027) -[reduce_map]-> BMxC
        origin_reduce_global_feature = self.reduce_map(
            origin_global_feature.reshape(B * M, -1))

        data['predictions']['origin_encode_feature'] = origin_encode_feature
        data['predictions'][
            'origin_reduce_global_feature'] = origin_reduce_global_feature

        if not self.infer:
            data = self.encodeCompleteShape(data)
        return data

    def encodeCompleteShape(self, data):
        # Bx#pointx3
        origin_point_array = data['inputs']['origin_point_array']

        # Bx#pointx3 -[feature_encoder]-> BxMxC and BxMx3
        origin_encode_feature, origin_coarse_point_cloud = self.feature_encoder(
            origin_point_array)

        B, M, C = origin_encode_feature.shape

        # BxMxC -[transpose]-> BxCxM -[increase_dim]-> Bx1024xM -[transpose]-> BxMx1024
        origin_points_feature = self.increase_dim(
            origin_encode_feature.transpose(1, 2)).transpose(1, 2)

        # BxMx1024 -[max]-> Bx1024
        origin_global_points_feature = torch.max(origin_points_feature,
                                                 dim=1)[0]

        # Bx1024 -[unsqueeze]-> Bx1x1024 -[expand]-> BxMx1024
        origin_replicate_global_points_feature = origin_global_points_feature.unsqueeze(
            -2).expand(-1, M, -1)

        # BxMx1024 + BxMxC + BxMx3 -[cat]-> BxMx(C+1027)
        origin_global_feature = torch.cat([
            origin_replicate_global_points_feature, origin_encode_feature,
            origin_coarse_point_cloud
        ],
                                          dim=-1)

        # BxMx(C+1027) -[reshape]-> BMx(C+1027) -[reduce_map]-> BMxC
        origin_reduce_global_feature = self.reduce_map(
            origin_global_feature.reshape(B * M, -1))

        data['predictions'][
            'complete_origin_encode_feature'] = origin_encode_feature
        data['predictions'][
            'complete_origin_reduce_global_feature'] = origin_reduce_global_feature
        return data

    def encodeRotateMatrix(self, data):
        # BMxC
        origin_reduce_global_feature = data['predictions'][
            'origin_reduce_global_feature']

        B = data['predictions']['origin_encode_feature'].shape[0]

        # BMxC -[reshape]-> BxMCx1 -[bbox_feature_encoder]-> BxCx1
        origin_rotate_feature = self.rotate_feature_encoder(
            origin_reduce_global_feature.reshape(B, -1, 1))

        # BxCx1 -[bbox_decoder]-> Bx6x1 -[reshape]-> Bx6
        rotation = self.rotate_decoder(origin_rotate_feature).reshape(B, -1)

        # Bx3x3
        rotate_matrix = compute_rotation_matrix_from_ortho6d(rotation)

        data['predictions']['origin_rotate_feature'] = origin_rotate_feature
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
        # BMxC
        origin_reduce_global_feature = data['predictions'][
            'complete_origin_reduce_global_feature']

        B = data['predictions']['complete_origin_encode_feature'].shape[0]

        # BMxC -[reshape]-> BxMCx1 -[bbox_feature_encoder]-> BxCx1
        origin_rotate_feature = self.rotate_feature_encoder(
            origin_reduce_global_feature.reshape(B, -1, 1))

        # BxCx1 -[bbox_decoder]-> Bx6x1 -[reshape]-> Bx6
        rotation = self.rotate_decoder(origin_rotate_feature).reshape(B, -1)

        # Bx3x3
        rotate_matrix = compute_rotation_matrix_from_ortho6d(rotation)

        data['predictions'][
            'complete_origin_rotate_feature'] = origin_rotate_feature
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

        data = setWeight(data, 'loss_rotate_matrix', 1000)
        #  data = setWeight(data, 'loss_geodesic', 1)

        data = setWeight(data, 'loss_complete_rotate_matrix', 1000)
        #  data = setWeight(data, 'loss_complete_geodesic', 1)
        return data

    def forward(self, data):
        data = self.encodeShape(data)

        data = self.encodeRotateMatrix(data)

        data = self.addWeight(data)

        if not self.infer:
            data = self.rotateBackByPredict(data)
        return data
