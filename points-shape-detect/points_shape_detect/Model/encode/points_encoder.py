#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from points_shape_detect.Method.weight import setWeight
from points_shape_detect.Model.encode.pc_transformer import PCTransformer


class PointsEncoder(nn.Module):

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

        self.infer = infer
        return

    @torch.no_grad()
    def moveToOriginWithGT(self, data):
        trans_point_array = data['inputs']['trans_point_array']
        # Bx#pointx3
        trans_query_point_array = data['inputs']['trans_query_point_array']

        trans_cad_point_array = data['inputs'].get('trans_cad_point_array')

        origin_points_list = []
        origin_query_points_list = []
        origin_bbox_list = []
        origin_center_list = []

        for i in range(trans_query_point_array.shape[0]):
            if trans_cad_point_array is not None:
                trans_points = trans_cad_point_array[i]
            else:
                trans_points = trans_point_array[i]
            trans_query_points = trans_query_point_array[i]
            trans_query_points_center = torch.mean(trans_query_points, 0)

            origin_points = trans_points - trans_query_points_center
            origin_query_points = trans_query_points - trans_query_points_center

            min_point = torch.min(origin_points, 0)[0]
            max_point = torch.max(origin_points, 0)[0]

            origin_bbox = torch.cat([min_point, max_point])
            min_max_point = origin_bbox.reshape(2, 3)
            origin_center = torch.mean(min_max_point, 0)

            origin_points_list.append(origin_points.unsqueeze(0))
            origin_query_points_list.append(origin_query_points.unsqueeze(0))
            origin_bbox_list.append(origin_bbox.unsqueeze(0))
            origin_center_list.append(origin_center.unsqueeze(0))

        origin_point_array = torch.cat(origin_points_list).detach()
        origin_query_point_array = torch.cat(origin_query_points_list).detach()
        origin_bbox = torch.cat(origin_bbox_list).detach()
        origin_center = torch.cat(origin_center_list).detach()

        data['inputs']['origin_point_array'] = origin_point_array
        data['inputs']['origin_query_point_array'] = origin_query_point_array
        data['inputs']['origin_bbox'] = origin_bbox
        data['inputs']['origin_center'] = origin_center
        return data

    @torch.no_grad()
    def moveToOrigin(self, data):
        if not self.infer:
            return self.moveToOriginWithGT(data)

        if 'origin_query_point_array' in data['inputs'].keys():
            return data

        # Bx#pointx3
        trans_query_point_array = data['inputs']['trans_query_point_array']

        origin_query_points_list = []

        for i in range(trans_query_point_array.shape[0]):
            trans_query_points = trans_query_point_array[i]
            trans_query_points_center = torch.mean(trans_query_points, 0)

            origin_query_points = trans_query_points - trans_query_points_center

            origin_query_points_list.append(origin_query_points.unsqueeze(0))

        origin_query_point_array = torch.cat(origin_query_points_list).detach()

        data['inputs']['origin_query_point_array'] = origin_query_point_array
        return data

    def encodeOriginPoints(self, data):
        # Bx#pointx3
        origin_query_point_array = data['inputs']['origin_query_point_array']

        # Bx#pointx3 -[feature_encoder]-> BxMxC and BxMx3
        origin_encode_feature, origin_coarse_point_cloud = self.feature_encoder(
            origin_query_point_array)

        data['predictions']['origin_encode_feature'] = origin_encode_feature
        data['predictions'][
            'origin_coarse_point_cloud'] = origin_coarse_point_cloud
        return data

    def embedOriginPointsFeature(self, data):
        # BxMxC
        origin_encode_feature = data['predictions']['origin_encode_feature']
        # BxMx3
        origin_coarse_point_cloud = data['predictions'][
            'origin_coarse_point_cloud']

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
            'origin_reduce_global_feature'] = origin_reduce_global_feature
        return data

    def forward(self, data):
        data = self.moveToOrigin(data)

        data = self.encodeOriginPoints(data)

        data = self.embedOriginPointsFeature(data)
        return data
