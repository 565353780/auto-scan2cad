#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from copy import deepcopy
from math import pi

import numpy as np
import open3d as o3d
import torch

from udf_generate.Config.sample import (SAMPLE_NUM, SAMPLE_POINT_CLOUD,
                                        SAMPLE_POINT_MATRIX)
from udf_generate.Method.paths import createFileFolder
from udf_generate.Method.samples import getMatrixFromArray


def getRad(angle):
    rad = angle * pi / 180.0
    return rad


def getAngle(rad):
    angle = rad * 180.0 / pi
    return angle


def loadMesh(mesh_file_path, trans_matrix=None):
    assert os.path.exists(mesh_file_path)

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)

    assert mesh is not None

    if trans_matrix is not None:
        mesh.transform(trans_matrix)
    return mesh


def getMeshBBox(mesh):
    bbox = mesh.get_axis_aligned_bounding_box()

    max_bound = bbox.get_max_bound()
    min_bound = bbox.get_min_bound()
    return min_bound, max_bound


def getMeshBBoxCenter(mesh):
    min_bound, max_bound = getMeshBBox(mesh)

    center = [(min_bound[0] + max_bound[0]) / 2.0,
              (min_bound[1] + max_bound[1]) / 2.0,
              (min_bound[2] + max_bound[2]) / 2.0]
    return center


def getMeshBBoxDiff(mesh):
    min_bound, max_bound = getMeshBBox(mesh)

    diff = [
        max_bound[0] - min_bound[0], max_bound[1] - min_bound[1],
        max_bound[2] - min_bound[2]
    ]
    return diff


def translateMesh(mesh, z_diff=0.0, x_diff=0.0, y_diff=0.0):
    if z_diff == 0.0 and x_diff == 0.0 and y_diff == 0.0:
        return True

    mesh.translate((z_diff, x_diff, y_diff))
    return True


def rotateMesh(mesh, z_angle=0.0, x_angle=0.0, y_angle=0.0):
    if z_angle == 0.0 and x_angle == 0.0 and y_angle == 0.0:
        return True

    z_rad = getRad(z_angle)
    x_rad = getRad(x_angle)
    y_rad = getRad(y_angle)

    R = mesh.get_rotation_matrix_from_xyz((z_rad, x_rad, y_rad))
    mesh.rotate(R, center=mesh.get_center())
    return True


def invRotateMesh(mesh, z_angle=0.0, x_angle=0.0, y_angle=0.0):
    if z_angle != 0.0:
        z_rad = getRad(z_angle)
        R = mesh.get_rotation_matrix_from_xyz((-z_rad, 0.0, 0.0))
        mesh.rotate(R, center=mesh.get_center())
    if x_angle != 0.0:
        x_rad = getRad(x_angle)
        R = mesh.get_rotation_matrix_from_xyz((0.0, -x_rad, 0.0))
        mesh.rotate(R, center=mesh.get_center())
    if y_angle != 0.0:
        y_rad = getRad(y_angle)
        R = mesh.get_rotation_matrix_from_xyz((0.0, 0.0, -y_rad))
        mesh.rotate(R, center=mesh.get_center())
    return True


def scaleMesh(mesh, scale=1):
    if scale == 1:
        return True

    mesh.scale(scale, center=mesh.get_center())
    return True


def normalizeMesh(mesh):
    diff = getMeshBBoxDiff(mesh)
    diff_max = max(diff)
    scaleMesh(mesh, 1.0 / diff_max)

    bbox_center = getMeshBBoxCenter(mesh)
    translateMesh(mesh, -bbox_center[0], -bbox_center[1], -bbox_center[2])
    return True


def getRaycastingScene(mesh):
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)
    return scene


def getPointDistListToMesh(scene, point_list):
    query_point_list = o3d.core.Tensor(point_list,
                                       dtype=o3d.core.Dtype.Float32)
    unsigned_distance_array = scene.compute_distance(query_point_list).numpy()
    return unsigned_distance_array


def getSignedPointDistListToMesh(scene, point_list):
    query_point_list = o3d.core.Tensor(point_list,
                                       dtype=o3d.core.Dtype.Float32)
    signed_distance_list = scene.compute_signed_distance(
        query_point_list).numpy()
    return signed_distance_list


def getUDF(mesh, z_angle=0.0, x_angle=0.0, y_angle=0.0):
    assert mesh is not None

    copy_mesh = deepcopy(mesh)
    rotateMesh(copy_mesh, z_angle, x_angle, y_angle)
    normalizeMesh(copy_mesh)

    scene = getRaycastingScene(copy_mesh)
    udf = getPointDistListToMesh(scene, SAMPLE_POINT_MATRIX)
    return udf


def getPointUDFTensor(point_array_tensor,
                      z_angle=0.0,
                      x_angle=0.0,
                      y_angle=0.0):
    device = point_array_tensor.device

    copy_point_array = deepcopy(point_array_tensor.detach().cpu().numpy())

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(copy_point_array)
    rotateMesh(pcd, z_angle, x_angle, y_angle)
    normalizeMesh(pcd)

    dist_array = np.array(SAMPLE_POINT_CLOUD.compute_point_cloud_distance(pcd))

    point_udf = getMatrixFromArray(dist_array, SAMPLE_NUM, 1)

    point_udf_tensor = torch.from_numpy(point_udf).to(torch.float32).to(device)
    return point_udf_tensor


def getPointUDF(point_array, z_angle=0.0, x_angle=0.0, y_angle=0.0):
    assert point_array is not None

    if isinstance(point_array, torch.Tensor):
        return getPointUDFTensor(point_array, z_angle, x_angle, y_angle)

    copy_point_array = deepcopy(point_array)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(copy_point_array)
    rotateMesh(pcd, z_angle, x_angle, y_angle)
    normalizeMesh(pcd)

    dist_array = np.array(SAMPLE_POINT_CLOUD.compute_point_cloud_distance(pcd))

    point_udf = getMatrixFromArray(dist_array, SAMPLE_NUM, 1)
    return point_udf


def saveUDF(udf, udf_save_file_path):
    assert udf is not None

    createFileFolder(udf_save_file_path)

    np.save(udf_save_file_path, udf)
    return True


def loadUDF(udf_file_path):
    assert os.path.exists(udf_file_path)

    udf = np.load(udf_file_path)

    assert udf is not None
    return udf


def getVisualUDF(udf, dist_max=0.02):
    point_list = []
    dist_list = []

    for z_idx in range(SAMPLE_POINT_MATRIX.shape[0]):
        for x_idx in range(SAMPLE_POINT_MATRIX.shape[1]):
            for y_idx in range(SAMPLE_POINT_MATRIX.shape[2]):
                udf_value = udf[z_idx][x_idx][y_idx]
                if udf_value > dist_max:
                    continue
                point_list.append(SAMPLE_POINT_MATRIX[z_idx][x_idx][y_idx])
                dist_list.append(udf_value)

    color_list = [[255, 0, 0] for _ in dist_list]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_list)
    pcd.colors = o3d.utility.Vector3dVector(color_list)
    return pcd
