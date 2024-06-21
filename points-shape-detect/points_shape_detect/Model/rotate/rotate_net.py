#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from udf_generate.Method.udfs import getPointUDF

from points_shape_detect.Model.resnet.resnet_encoder import ResNetEncoder
from points_shape_detect.Model.resnet.resnet_decoder import ResNetDecoder

from points_shape_detect.Method.trans import transPointArray
from points_shape_detect.Method.weight import setWeight


class RotateNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_dim = 1024

        self.shape_encoder = ResNetEncoder(self.feature_dim)

        self.euler_angle_encoder = nn.Sequential(
            nn.Conv1d(self.feature_dim, 3, 1),
            nn.LeakyReLU(negative_slope=0.2), nn.Conv1d(3, 3, 1))

        self.shape_decoder = ResNetDecoder(self.shape_encoder.feats)

        self.l1_loss = nn.SmoothL1Loss()
        return

    @torch.no_grad()
    def generateUDFWithGT(self, data):
        trans_point_array = data['inputs']['trans_point_array']
        trans_query_point_array = data['inputs']['trans_query_point_array']
        gt_euler_angle_inv = data['inputs']['euler_angle_inv']

        device = trans_query_point_array.device

        trans_udf_list = []
        trans_query_udf_list = []
        rotate_back_udf_list = []
        rotate_back_query_udf_list = []

        translate = torch.tensor([0.0, 0.0, 0.0],
                                 dtype=torch.float32).to(device)
        scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)
        for i in range(trans_query_point_array.shape[0]):
            trans_points = trans_point_array[i]
            trans_query_points = trans_query_point_array[i]
            euler_angle = gt_euler_angle_inv[i]

            rotate_back_points = transPointArray(trans_points, translate,
                                                 euler_angle, scale, True,
                                                 translate)
            rotate_back_query_points = transPointArray(trans_query_points,
                                                       translate, euler_angle,
                                                       scale, True, translate)
            trans_udf = getPointUDF(trans_points)
            trans_query_udf = getPointUDF(trans_query_points)
            rotate_back_udf = getPointUDF(rotate_back_points)
            rotate_back_query_udf = getPointUDF(rotate_back_query_points)

            trans_udf_list.append(trans_udf.unsqueeze(0))
            trans_query_udf_list.append(trans_query_udf.unsqueeze(0))
            rotate_back_udf_list.append(rotate_back_udf.unsqueeze(0))
            rotate_back_query_udf_list.append(
                rotate_back_query_udf.unsqueeze(0))

        trans_udf = torch.cat(trans_udf_list).detach()
        trans_query_udf = torch.cat(trans_query_udf_list).detach()
        rotate_back_udf = torch.cat(rotate_back_udf_list).detach()
        rotate_back_query_udf = torch.cat(rotate_back_query_udf_list).detach()

        data['predictions']['trans_udf'] = trans_udf
        data['predictions']['trans_query_udf'] = trans_query_udf
        data['predictions']['rotate_back_udf'] = rotate_back_udf
        data['predictions']['rotate_back_query_udf'] = rotate_back_query_udf
        return data

    @torch.no_grad()
    def generateUDF(self, data):
        if self.training:
            return self.generateUDFWithGT(data)

        trans_query_point_array = data['inputs']['trans_query_point_array']

        device = trans_query_point_array.device

        trans_query_udf_list = []

        translate = torch.tensor([0.0, 0.0, 0.0],
                                 dtype=torch.float32).to(device)
        scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(device)
        for i in range(trans_query_point_array.shape[0]):
            trans_query_points = trans_query_point_array[i]

            trans_query_udf = getPointUDF(trans_query_points)

            trans_query_udf_list.append(trans_query_udf.unsqueeze(0))

        trans_query_udf = torch.cat(trans_query_udf_list).detach()

        data['predictions']['trans_query_udf'] = trans_query_udf
        return data

    def encodeQueryUDFWithGT(self, data):
        # Bx32x32
        trans_udf = data['predictions']['trans_udf']
        # Bx32x32
        trans_query_udf = data['predictions']['trans_query_udf']

        trans_shape_code = self.shape_encoder(trans_udf.unsqueeze(1))
        trans_query_shape_code = self.shape_encoder(
            trans_query_udf.unsqueeze(1))

        data['predictions']['trans_shape_code'] = trans_shape_code
        data['predictions']['trans_query_shape_code'] = trans_query_shape_code
        return data

    def encodeQueryUDF(self, data):
        if self.training:
            return self.encodeQueryUDFWithGT(data)

        # Bx32x32
        trans_query_udf = data['predictions']['trans_query_udf']

        trans_query_shape_code = self.shape_encoder(
            trans_query_udf.unsqueeze(1))

        data['predictions']['trans_query_shape_code'] = trans_query_shape_code
        return data

    def encodeRotateWithGT(self, data):
        # Bxself.feature_dim
        trans_shape_code = data['predictions']['trans_shape_code']
        # Bxself.feature_dim
        trans_query_shape_code = data['predictions']['trans_query_shape_code']

        B, C = data['predictions']['trans_query_shape_code'].shape

        trans_euler_angle_inv = self.euler_angle_encoder(
            trans_shape_code.unsqueeze(-1)).reshape(B, -1)
        trans_query_euler_angle_inv = self.euler_angle_encoder(
            trans_query_shape_code.unsqueeze(-1)).reshape(B, -1)

        data['predictions']['trans_euler_angle_inv'] = trans_euler_angle_inv
        data['predictions'][
            'trans_query_euler_angle_inv'] = trans_query_euler_angle_inv

        if self.training:
            data = self.lossRotate(data)
        return data

    def encodeRotate(self, data):
        if self.training:
            return self.encodeRotateWithGT(data)

        # Bxself.feature_dim
        trans_query_shape_code = data['predictions']['trans_query_shape_code']

        B, C = data['predictions']['trans_query_shape_code'].shape

        trans_query_euler_angle_inv = self.euler_angle_encoder(
            trans_query_shape_code.unsqueeze(-1)).reshape(B, -1)

        data['predictions'][
            'trans_query_euler_angle_inv'] = trans_query_euler_angle_inv
        return data

    def lossRotate(self, data):
        trans_euler_angle_inv = data['predictions']['trans_euler_angle_inv']
        trans_query_euler_angle_inv = data['predictions'][
            'trans_query_euler_angle_inv']
        gt_euler_angle_inv = data['inputs']['euler_angle_inv']

        loss_trans_euler_angle_inv = self.l1_loss(trans_euler_angle_inv,
                                                  gt_euler_angle_inv)
        loss_trans_query_euler_angle_inv = self.l1_loss(
            trans_query_euler_angle_inv, gt_euler_angle_inv)
        loss_partial_complete_euler_angle_inv_diff = self.l1_loss(
            trans_euler_angle_inv, trans_query_euler_angle_inv)

        data['losses'][
            'loss_trans_euler_angle_inv'] = loss_trans_euler_angle_inv
        data['losses'][
            'loss_trans_query_euler_angle_inv'] = loss_trans_query_euler_angle_inv
        data['losses'][
            'loss_partial_complete_euler_angle_inv_diff'] = loss_partial_complete_euler_angle_inv_diff
        return data

    def decodeShapeCodeWithGT(self, data):
        trans_shape_code = data['predictions']['trans_shape_code']
        trans_query_shape_code = data['predictions']['trans_query_shape_code']

        decode_trans_udf = torch.squeeze(self.shape_decoder(trans_shape_code),
                                         1)
        decode_trans_query_udf = torch.squeeze(
            self.shape_decoder(trans_query_shape_code), 1)

        data['predictions']['decode_trans_udf'] = decode_trans_udf
        data['predictions']['decode_trans_query_udf'] = decode_trans_query_udf

        self.lossUDF(data)
        return data

    def decodeShapeCode(self, data):
        if self.training:
            return self.decodeShapeCodeWithGT(data)

        trans_query_shape_code = data['predictions']['trans_query_shape_code']

        decode_trans_query_udf = torch.squeeze(
            self.shape_decoder(trans_query_shape_code), 1)

        data['predictions']['decode_trans_query_udf'] = decode_trans_query_udf
        return data

    def lossUDF(self, data):
        rotate_back_udf = data['predictions']['rotate_back_udf']
        rotate_back_query_udf = data['predictions']['rotate_back_query_udf']
        decode_trans_udf = data['predictions']['decode_trans_udf']
        decode_trans_query_udf = data['predictions']['decode_trans_query_udf']

        loss_decode_trans_udf = self.l1_loss(decode_trans_udf, rotate_back_udf)
        loss_decode_trans_query_udf = self.l1_loss(decode_trans_query_udf,
                                                   rotate_back_query_udf)

        data['losses']['loss_decode_trans_udf'] = loss_decode_trans_udf
        data['losses'][
            'loss_decode_trans_query_udf'] = loss_decode_trans_query_udf
        return data

    def addWeight(self, data):
        if not self.training:
            return data

        setWeight(data, 'loss_trans_euler_angle_inv', 1)
        setWeight(data, 'loss_trans_query_euler_angle_inv', 1)
        setWeight(data, 'loss_partial_complete_euler_angle_inv_diff', 1)

        setWeight(data, 'loss_decode_trans_udf', 1000)
        setWeight(data, 'loss_decode_trans_query_udf', 1000)
        return data

    def forward(self, data):
        data = self.generateUDF(data)

        data = self.encodeQueryUDF(data)

        data = self.encodeRotate(data)

        data = self.rotateBackPoints(data)

        data = self.decodeShapeCode(data)
        return data
