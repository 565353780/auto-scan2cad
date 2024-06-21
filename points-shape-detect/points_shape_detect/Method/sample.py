#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import torch.nn.functional as F

from pointnet2_ops import pointnet2_utils


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    if data.shape[1] == number:
        return data

    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(
        data.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                              2).contiguous()
    return fps_data


def seprate_point_cloud(xyz, crop, fixed_points=None, padding_zeros=False):
    '''
     seprate point cloud: usage : using to generate the incomplete point cloud with a setted number.
    '''
    assert 2 <= len(xyz.shape) <= 3
    if len(xyz.shape) == 2:
        n, c = xyz.shape
    else:
        _, n, c = xyz.shape
    assert c == 3

    if crop == 0:
        return xyz, None
    if isinstance(crop, list):
        num_crop = random.randint(int(n * crop[0]), int(n * crop[1]))
        if num_crop == 0:
            return xyz, None

    ndarray_input = False
    if isinstance(xyz, np.ndarray):
        ndarray_input = True
        xyz = torch.from_numpy(xyz.reshape(1, -1, 3)).float().cuda()

    INPUT = []
    CROP = []
    for points in xyz:
        if not isinstance(crop, list):
            num_crop = int(n * crop)

        points = points.unsqueeze(0)

        if fixed_points is None:
            center = F.normalize(torch.randn(1, 1, 3), p=2, dim=-1).cuda()
        else:
            if isinstance(fixed_points, list):
                fixed_point = random.sample(fixed_points, 1)[0]
            else:
                fixed_point = fixed_points
            center = fixed_point.reshape(1, 1, 3).cuda()

        distance_matrix = torch.norm(center.unsqueeze(2) - points.unsqueeze(1),
                                     p=2,
                                     dim=-1)  # 1 1 2048

        idx = torch.argsort(distance_matrix, dim=-1,
                            descending=False)[0, 0]  # 2048

        if padding_zeros:
            input_data = points.clone()
            input_data[0, idx[:num_crop]] = input_data[0, idx[:num_crop]] * 0

        else:
            input_data = points.clone()[0,
                                        idx[num_crop:]].unsqueeze(0)  # 1 N 3

        crop_data = points.clone()[0, idx[:num_crop]].unsqueeze(0)

        if isinstance(crop, list):
            INPUT.append(fps(input_data, 2048))
            CROP.append(fps(crop_data, 2048))
        else:
            INPUT.append(input_data)
            CROP.append(crop_data)

    input_data = torch.cat(INPUT, dim=0)  # B N 3
    crop_data = torch.cat(CROP, dim=0)  # B M 3

    if ndarray_input:
        return input_data.contiguous().cpu().numpy().reshape(
            -1, 3), crop_data.contiguous().cpu().numpy().reshape(-1, 3)

    return input_data.contiguous(), crop_data.contiguous()
