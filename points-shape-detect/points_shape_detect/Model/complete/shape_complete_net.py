#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from points_shape_detect.Lib.chamfer_dist import WeightedChamferDistanceL2
from points_shape_detect.Method.sample import fps
from points_shape_detect.Method.weight import setWeight
from points_shape_detect.Model.complete.fold import Fold


class ShapeCompleteNet(nn.Module):

    def __init__(self, infer=False):
        super().__init__()
        #M
        self.num_query = 96
        #C
        self.trans_dim = 384
        self.num_pred = 6144

        self.fold_step = int(pow(self.num_pred // self.num_query, 0.5) + 0.5)

        self.foldingnet = Fold(self.trans_dim,
                               step=self.fold_step,
                               hidden_dim=256)  # rebuild a cluster point

        self.loss_func = WeightedChamferDistanceL2()

        self.infer = infer
        return

    def decodeOriginPatchPoints(self, data):
        # BMxC
        origin_reduce_global_feature = data['predictions'][
            'origin_reduce_global_feature']
        # BxMx3
        origin_coarse_point_cloud = data['predictions'][
            'origin_coarse_point_cloud']

        B, M, C = data['predictions']['origin_encode_feature'].shape

        # BMxC -[foldingnet]-> BMx3xS -[reshape]-> BxMx3xS
        origin_relative_patch_points = self.foldingnet(
            origin_reduce_global_feature).reshape(B, M, 3, -1)

        # BxMx3xS + BxMx3x1 = BxMx3xS -[transpose]-> BxMxSx3
        origin_rebuild_patch_points = (
            origin_relative_patch_points +
            origin_coarse_point_cloud.unsqueeze(-1)).transpose(2, 3)

        # BxMxSx3 -[reshape]-> BxMSx3
        origin_rebuild_points = origin_rebuild_patch_points.reshape(B, -1, 3)

        data['predictions'][
            'origin_rebuild_patch_points'] = origin_rebuild_patch_points
        data['predictions']['origin_rebuild_points'] = origin_rebuild_points
        return data

    def embedOriginPoints(self, data):
        # Bx#pointx3
        origin_query_point_array = data['inputs']['origin_query_point_array']
        # BxMx3
        origin_coarse_point_cloud = data['predictions'][
            'origin_coarse_point_cloud']
        # BxMSx3
        origin_rebuild_points = data['predictions']['origin_rebuild_points']

        # Bx#pointx3 -[fps]-> BxMx3
        fps_origin_query_point_array = fps(origin_query_point_array,
                                           self.num_query)

        # BxMx3 + BxMx3 -[cat]-> Bx2Mx3
        origin_coarse_points = torch.cat(
            [origin_coarse_point_cloud, fps_origin_query_point_array],
            dim=1).contiguous()

        # BxMSx3 + Bx#pointx3 -[cat]-> Bx(MS+#point)x3
        origin_dense_points = torch.cat(
            [origin_rebuild_points, origin_query_point_array],
            dim=1).contiguous()

        data['predictions']['origin_coarse_points'] = origin_coarse_points
        data['predictions']['origin_dense_points'] = origin_dense_points

        if not self.infer:
            data = self.lossOriginComplete(data)
        return data

    def lossOriginComplete(self, data):
        origin_query_point_array = data['inputs']['origin_query_point_array']
        origin_point_array = data['inputs']['origin_point_array']
        origin_coarse_points = data['predictions']['origin_coarse_points']
        origin_dense_points = data['predictions']['origin_dense_points']

        loss_origin_coarse = self.loss_func(origin_query_point_array,
                                            origin_coarse_points,
                                            origin_point_array)
        loss_origin_fine = self.loss_func(origin_query_point_array,
                                          origin_dense_points,
                                          origin_point_array)

        data['losses']['loss_origin_coarse'] = loss_origin_coarse
        data['losses']['loss_origin_fine'] = loss_origin_fine
        return data

    def addWeight(self, data):
        if self.infer:
            return data

        data = setWeight(data, 'loss_origin_coarse', 10000)
        data = setWeight(data, 'loss_origin_fine', 10000)
        return data

    def forward(self, data):
        data = self.decodeOriginPatchPoints(data)

        data = self.embedOriginPoints(data)

        data = self.addWeight(data)
        return data
