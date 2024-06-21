#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from points_shape_detect.Loss.ious import IoULoss
from points_shape_detect.Method.weight import setWeight


class BBoxNet(nn.Module):

    def __init__(self, infer=False):
        super().__init__()
        #M
        self.num_query = 96
        #C
        self.trans_dim = 384

        self.bbox_feature_encoder = nn.Sequential(
            nn.Conv1d(self.num_query * self.trans_dim, self.trans_dim, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1))

        self.bbox_decoder = nn.Sequential(nn.Conv1d(self.trans_dim, 6, 1),
                                          nn.LeakyReLU(negative_slope=0.2),
                                          nn.Conv1d(6, 6, 1))

        self.center_decoder = nn.Sequential(nn.Conv1d(self.trans_dim, 3, 1),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Conv1d(3, 3, 1))

        self.l1_loss = nn.SmoothL1Loss()

        self.infer = infer
        return

    def encodeOriginBBoxFeature(self, data):
        # BMxC
        origin_reduce_global_feature = data['predictions'][
            'origin_reduce_global_feature']

        B = data['predictions']['origin_encode_feature'].shape[0]

        # BMxC -[reshape]-> BxMCx1 -[bbox_feature_encoder]-> BxCx1
        origin_bbox_feature = self.bbox_feature_encoder(
            origin_reduce_global_feature.reshape(B, -1, 1))

        data['predictions']['origin_bbox_feature'] = origin_bbox_feature
        return data

    def encodeOriginBBox(self, data):
        # BxCx1
        origin_bbox_feature = data['predictions']['origin_bbox_feature']

        B = origin_bbox_feature.shape[0]

        # BxCx1 -[bbox_decoder]-> Bx6x1 -[reshape]-> Bx6
        origin_bbox = self.bbox_decoder(origin_bbox_feature).reshape(B, -1)

        # BxCx1 -[center_decoder]-> Bx3x1 -[reshape]-> Bx3
        origin_center = self.center_decoder(origin_bbox_feature).reshape(B, -1)

        data['predictions']['origin_bbox'] = origin_bbox
        data['predictions']['origin_center'] = origin_center

        if not self.infer:
            data = self.lossOriginBBox(data)
        return data

    def lossOriginBBox(self, data):
        origin_bbox = data['predictions']['origin_bbox']
        origin_center = data['predictions']['origin_center']
        gt_origin_bbox = data['inputs']['origin_bbox']
        gt_origin_center = data['inputs']['origin_center']

        loss_origin_bbox_l1 = self.l1_loss(origin_bbox, gt_origin_bbox)
        loss_origin_center_l1 = self.l1_loss(origin_center, gt_origin_center)
        loss_origin_bbox_eiou = torch.mean(
            IoULoss.EIoU(origin_bbox, gt_origin_bbox))

        data['losses']['loss_origin_bbox_l1'] = loss_origin_bbox_l1
        data['losses']['loss_origin_center_l1'] = loss_origin_center_l1
        data['losses']['loss_origin_bbox_eiou'] = loss_origin_bbox_eiou
        return data

    def addWeight(self, data):
        if self.infer:
            return data

        data = setWeight(data, 'loss_origin_bbox_l1', 1000)
        data = setWeight(data, 'loss_origin_center_l1', 1000)
        data = setWeight(data, 'loss_origin_bbox_eiou', 100, max_value=100)
        return data

    def forward(self, data):
        data = self.encodeOriginBBoxFeature(data)

        data = self.encodeOriginBBox(data)

        data = self.addWeight(data)
        return data
