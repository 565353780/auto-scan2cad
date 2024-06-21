#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
import torch

from points_shape_detect.Method.matrix import getRotateMatrix


def getBatchResult(func, batch_data):
    batch_result = torch.cat(
        [func(batch_data[i]).unsqueeze(0) for i in range(batch_data.shape[0])])
    return batch_result


def getBatchResultWithPair(func, batch_data, pair_batch_data=None):
    if cad_batch_data is None:
        return getBatchResult(func, batch_data)

    batch_result_list = [[func(batch_data[i], pair_batch_data[i])]
                         for i in range(batch_data.shape[0])]

    batch_result = torch.cat([
        batch_result_list[i][0].unsqueeze(0)
        for i in range(batch_data.shape[0])
    ])
    pair_batch_result = torch.cat([
        batch_result_list[i][1].unsqueeze(0)
        for i in range(batch_data.shape[0])
    ])
    return batch_result, pair_batch_result


def normalizePointArrayTensor(point_array_tensor,
                              pair_point_array_tensor=None):
    min_point_tensor = torch.min(point_array_tensor, 0)[0]
    max_point_tensor = torch.max(point_array_tensor, 0)[0]
    min_max_point_tensor = torch.cat([min_point_tensor,
                                      max_point_tensor]).reshape(2, 3)
    center = torch.mean(min_max_point_tensor, 0)

    origin_point_array_tensor = point_array_tensor - center

    max_bbox_length = torch.max(max_point_tensor - min_point_tensor)
    normalize_point_array_tensor = origin_point_array_tensor / max_bbox_length

    if pair_point_array_tensor is None:
        return normalize_point_array_tensor

    origin_pair_point_array_tensor = pair_point_array_tensor - center
    normalize_pair_point_array_tensor = origin_pair_point_array_tensor / max_bbox_length
    return normalize_point_array_tensor, normalize_pair_point_array_tensor


def normalizePointArray(point_array, pair_point_array=None):
    if isinstance(point_array, torch.Tensor):
        assert 2 <= len(point_array.shape) <= 3
        if len(point_array.shape) == 2:
            return normalizePointArrayTensor(point_array, pair_point_array)
        else:
            return getBatchResultWithPair(normalizePointArrayTensor,
                                          point_array, pair_point_array)

    min_point = np.min(point_array, axis=0)
    max_point = np.max(point_array, axis=0)
    center = np.mean([min_point, max_point], axis=0)

    origin_point_array = point_array - center

    max_bbox_length = np.max(max_point - min_point)
    normalize_point_array = origin_point_array / max_bbox_length

    if pair_point_array is None:
        return normalize_point_array

    origin_pair_point_array = pair_point_array - center
    normalize_pair_point_array = origin_pair_point_array / max_bbox_length
    return normalize_point_array, normalize_pair_point_array


def getInverseTrans(translate, euler_angle, scale):
    translate_inv = -1.0 * translate
    euler_angle_inv = -1.0 * euler_angle
    scale_inv = 1.0 / scale
    return translate_inv, euler_angle_inv, scale_inv


def transPointArrayTensor(point_array_tensor,
                          translate,
                          euler_angle,
                          scale,
                          is_inverse=False,
                          center=None):
    if center is None:
        center = torch.mean(point_array_tensor, 0)

    origin_point_array_tensor = point_array_tensor - center

    rotate_matrix = getRotateMatrix(euler_angle, is_inverse)

    if is_inverse:
        trans_origin_point_array_tensor = torch.matmul(
            origin_point_array_tensor, rotate_matrix)
        final_origin_point_array_tensor = trans_origin_point_array_tensor * scale
    else:
        scale_origin_point_array_tensor = origin_point_array_tensor * scale
        final_origin_point_array_tensor = torch.matmul(
            scale_origin_point_array_tensor, rotate_matrix)

    trans_point_array_tensor = final_origin_point_array_tensor + center + translate
    return trans_point_array_tensor


def transPointArray(point_array,
                    translate,
                    euler_angle,
                    scale,
                    is_inverse=False,
                    center=None):
    if isinstance(point_array, torch.Tensor):
        return transPointArrayTensor(point_array, translate, euler_angle,
                                     scale, is_inverse, center)

    if center is None:
        center = np.mean(point_array, axis=0)

    origin_point_array = point_array - center

    rotate_matrix = getRotateMatrix(euler_angle, is_inverse)

    if is_inverse:
        trans_origin_point_array = origin_point_array @ rotate_matrix
        final_origin_point_array = trans_origin_point_array * scale
    else:
        scale_origin_point_array = origin_point_array * scale
        final_origin_point_array = scale_origin_point_array @ rotate_matrix

    trans_point_array = final_origin_point_array + center + translate
    return trans_point_array


def randomTransPointArrayTensor(point_array_tensor, need_trans=False):
    device = point_array_tensor.device

    translate = (torch.rand(3) - 0.5).to(device)
    euler_angle = ((torch.rand(3) - 0.5) * 360.0).to(device)
    scale = (torch.rand(3) + 0.5).to(device)

    trans_point_array_tensor = transPointArray(point_array_tensor, translate,
                                               euler_angle, scale)
    if need_trans:
        return trans_point_array_tensor, translate, euler_angle, scale
    return trans_point_array_tensor


def randomTransPointArray(point_array, need_trans=False):
    if isinstance(point_array, torch.Tensor):
        return randomTransPointArrayTensor(point_array, need_trans)

    translate = np.random.rand(3) - 0.5
    euler_angle = (np.random.rand(3) - 0.5) * 360.0
    scale = np.random.rand(3) + 0.5

    trans_point_array = transPointArray(point_array, translate, euler_angle,
                                        scale)
    if need_trans:
        return trans_point_array, translate, euler_angle, scale
    return trans_point_array


def moveToOrigin(point_array):
    min_point = np.min(point_array, axis=0)
    max_point = np.max(point_array, axis=0)
    center = np.mean([min_point, max_point], axis=0)
    return point_array - center


def moveToMeanPoint(point_array):
    mean_xyz = np.array([np.mean(point_array[:, i]) for i in range(3)])
    return point_array - mean_xyz
