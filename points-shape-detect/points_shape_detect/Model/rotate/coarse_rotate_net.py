#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from points_shape_detect.Method.weight import setWeight


class CoarseRotateNet(nn.Module):

    def __init__(self):
        super().__init__()
        #M
        self.num_query = 96
        #C
        self.trans_dim = 384
        #R
        self.coarse_rotate_cls_num = 18

        self.coarse_rotate_feature_encoder = nn.Sequential(
            nn.Conv1d(self.num_query * self.trans_dim, self.trans_dim, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1))

        self.x_coarse_rotate_decoder = nn.Sequential(
            nn.Conv1d(self.trans_dim, self.coarse_rotate_cls_num, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.coarse_rotate_cls_num, self.coarse_rotate_cls_num,
                      1))
        self.y_coarse_rotate_decoder = nn.Sequential(
            nn.Conv1d(self.trans_dim, self.coarse_rotate_cls_num, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.coarse_rotate_cls_num, self.coarse_rotate_cls_num,
                      1))
        self.z_coarse_rotate_decoder = nn.Sequential(
            nn.Conv1d(self.trans_dim, self.coarse_rotate_cls_num, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.coarse_rotate_cls_num, self.coarse_rotate_cls_num,
                      1))

        self.cls_loss = nn.CrossEntropyLoss()
        return

    @torch.no_grad()
    def createCoarseRotateCls(self, data):
        # Bx3
        euler_angle = data['inputs']['euler_angle']

        device = euler_angle.device

        x_coarse_rotate_cls_list = []
        y_coarse_rotate_cls_list = []
        z_coarse_rotate_cls_list = []

        for current_euler_angle in euler_angle:
            x_rotate_angle, y_rotate_angle, z_rotate_angle = current_euler_angle
            while x_rotate_angle < 0:
                x_rotate_angle = x_rotate_angle + 360.0
            while x_rotate_angle >= 360.0:
                x_rotate_angle = x_rotate_angle - 360.0
            while y_rotate_angle < 0:
                y_rotate_angle = y_rotate_angle + 360.0
            while y_rotate_angle >= 360.0:
                y_rotate_angle = y_rotate_angle - 360.0
            while z_rotate_angle < 0:
                z_rotate_angle = z_rotate_angle + 360.0
            while z_rotate_angle >= 360.0:
                z_rotate_angle = z_rotate_angle - 360.0

            x_rotate_cls_idx = int(
                (x_rotate_angle * self.coarse_rotate_cls_num /
                 360.0).floor().item())
            y_rotate_cls_idx = int(
                (y_rotate_angle * self.coarse_rotate_cls_num /
                 360.0).floor().item())
            z_rotate_cls_idx = int(
                (z_rotate_angle * self.coarse_rotate_cls_num /
                 360.0).floor().item())

            x_rotate_cls = torch.zeros(self.coarse_rotate_cls_num,
                                       dtype=torch.float32).to(device)
            x_rotate_cls[x_rotate_cls_idx] = 1.0
            y_rotate_cls = torch.zeros(self.coarse_rotate_cls_num,
                                       dtype=torch.float32).to(device)
            y_rotate_cls[y_rotate_cls_idx] = 1.0
            z_rotate_cls = torch.zeros(self.coarse_rotate_cls_num,
                                       dtype=torch.float32).to(device)
            z_rotate_cls[z_rotate_cls_idx] = 1.0

            x_coarse_rotate_cls_list.append(x_rotate_cls.unsqueeze(0))
            y_coarse_rotate_cls_list.append(y_rotate_cls.unsqueeze(0))
            z_coarse_rotate_cls_list.append(z_rotate_cls.unsqueeze(0))

        x_coarse_rotate_cls = torch.cat(x_coarse_rotate_cls_list).detach()
        y_coarse_rotate_cls = torch.cat(y_coarse_rotate_cls_list).detach()
        z_coarse_rotate_cls = torch.cat(z_coarse_rotate_cls_list).detach()

        data['inputs']['x_coarse_rotate_cls'] = x_coarse_rotate_cls
        data['inputs']['y_coarse_rotate_cls'] = y_coarse_rotate_cls
        data['inputs']['z_coarse_rotate_cls'] = z_coarse_rotate_cls
        return data

    def encodeCoarseRotate(self, data):
        # BMxC
        origin_reduce_global_feature = data['predictions'][
            'origin_reduce_global_feature']

        B = data['predictions']['origin_encode_feature'].shape[0]

        # BMxC -[reshape]-> BxMCx1 -[bbox_feature_encoder]-> BxCx1
        origin_coarse_rotate_feature = self.coarse_rotate_feature_encoder(
            origin_reduce_global_feature.reshape(B, -1, 1))

        # BxCx1 -[x_coarse_rotate_decoder]-> BxRx1 -[reshape]-> BxR
        x_coarse_rotate_cls = self.x_coarse_rotate_decoder(
            origin_coarse_rotate_feature).reshape(B, -1)

        # BxCx1 -[y_coarse_rotate_decoder]-> BxRx1 -[reshape]-> BxR
        y_coarse_rotate_cls = self.y_coarse_rotate_decoder(
            origin_coarse_rotate_feature).reshape(B, -1)

        # BxCx1 -[z_coarse_rotate_decoder]-> BxRx1 -[reshape]-> BxR
        z_coarse_rotate_cls = self.z_coarse_rotate_decoder(
            origin_coarse_rotate_feature).reshape(B, -1)

        data['predictions']['x_coarse_rotate_cls'] = x_coarse_rotate_cls
        data['predictions']['y_coarse_rotate_cls'] = y_coarse_rotate_cls
        data['predictions']['z_coarse_rotate_cls'] = z_coarse_rotate_cls

        #  if self.training:
        data = self.lossCoarseRotate(data)
        return data

    def lossCoarseRotate(self, data):
        x_coarse_rotate_cls = data['predictions']['x_coarse_rotate_cls']
        y_coarse_rotate_cls = data['predictions']['y_coarse_rotate_cls']
        z_coarse_rotate_cls = data['predictions']['z_coarse_rotate_cls']
        gt_x_coarse_rotate_cls = data['inputs']['x_coarse_rotate_cls']
        gt_y_coarse_rotate_cls = data['inputs']['y_coarse_rotate_cls']
        gt_z_coarse_rotate_cls = data['inputs']['z_coarse_rotate_cls']

        loss_x_coarse_rotate_cls = self.cls_loss(x_coarse_rotate_cls,
                                                 gt_x_coarse_rotate_cls)
        loss_y_coarse_rotate_cls = self.cls_loss(y_coarse_rotate_cls,
                                                 gt_y_coarse_rotate_cls)
        loss_z_coarse_rotate_cls = self.cls_loss(z_coarse_rotate_cls,
                                                 gt_z_coarse_rotate_cls)

        data['losses']['loss_x_coarse_rotate_cls'] = loss_x_coarse_rotate_cls
        data['losses']['loss_y_coarse_rotate_cls'] = loss_y_coarse_rotate_cls
        data['losses']['loss_z_coarse_rotate_cls'] = loss_z_coarse_rotate_cls
        return data

    def addWeight(self, data):
        #  if not self.training:
        #  return data

        data = setWeight(data, 'loss_x_coarse_rotate_cls', 100)
        data = setWeight(data, 'loss_y_coarse_rotate_cls', 100)
        data = setWeight(data, 'loss_z_coarse_rotate_cls', 100)
        return data

    def forward(self, data):
        data = self.createCoarseRotateCls(data)

        data = self.encodeCoarseRotate(data)

        data = self.addWeight(data)
        return data
