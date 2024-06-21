#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../udf-generate")

import os
import numpy as np
import open3d as o3d

from points_shape_detect.Method.sample import seprate_point_cloud
from points_shape_detect.Method.trans import normalizePointArray, moveToOrigin
from points_shape_detect.Method.render import \
    renderPointArrayWithUnitBBox, \
    renderRebuildPatchPoints, \
    renderPredictBBox, renderRotateBackPoints
from points_shape_detect.Method.rotate import get_sampled_rotation_matrices_by_axisAngle

from points_shape_detect.Module.rotate_detector import RotateDetector


def demo():
    model_file_path = "./output/pretrained_rotate/model_best.pth"
    npy_file_path = "/home/chli/chLi/PoinTr/ShapeNet55/shapenet_pc/" + \
        "04090263-2eb7e88f5355630962a5697e98a94be.npy"

    rotate_detector = RotateDetector(model_file_path)

    points = np.load(npy_file_path)
    points = normalizePointArray(points)

    for i in range(1000):
        partial, _ = seprate_point_cloud(points, 0.5)
        partial = moveToOrigin(partial)

        rotate_matrix = get_sampled_rotation_matrices_by_axisAngle(
            1).cpu().numpy()[0]
        partial = np.dot(partial, rotate_matrix)

        renderPointArrayWithUnitBBox(partial)

        data = rotate_detector.detectPointArray(partial)

        print(data['inputs'].keys())
        print(data['predictions'].keys())
        renderRotateBackPoints(data)
    return True


def demo_mesh():
    model_file_path = "./output/pretrained_rotate/model_best.pth"
    shapenet_model_file_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/" + \
        "02691156/1a04e3eab45ca15dd86060f189eb133" + \
        "/models/model_normalized.obj"

    rotate_detector = RotateDetector(model_file_path)

    assert os.path.exists(shapenet_model_file_path)
    mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
    pcd = mesh.sample_points_uniformly(8192)
    points = np.array(pcd.points)
    points = normalizePointArray(points)

    partial, _ = seprate_point_cloud(points, 0.5)
    partial = moveToOrigin(partial)
    rotate_matrix = get_sampled_rotation_matrices_by_axisAngle(
        1).cpu().numpy()[0]
    partial = np.dot(partial, rotate_matrix)

    renderPointArrayWithUnitBBox(partial)

    data = rotate_detector.detectPointArray(partial)

    print(data['predictions'].keys())
    renderRotateBackPoints(data)
    return True
