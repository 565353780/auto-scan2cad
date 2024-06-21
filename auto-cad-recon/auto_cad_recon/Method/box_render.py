#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

from global_pose_refine.Method.render import getOBBPCD
from points_shape_detect.Method.trans import getInverseTrans, transPointArray

from auto_cad_recon.Method.bbox import getOBBFromABB
from auto_cad_recon.Method.trans import transPoints


def getABBPCD(bbox, color=None):
    obb = getOBBFromABB(bbox)
    return getOBBPCD(obb, color)


def getGTOBBList(data_list, color_map=None, translate=[0, 0, 0]):
    translate = np.array(translate, dtype=float)

    obb_list = []

    if color_map is None:
        color_map = []
        for i in range(100):
            random_color = np.random.rand(3) * 255.0
            color_map.append(random_color)
    color_map = np.array(color_map, dtype=float)

    for i, data in enumerate(data_list):
        shapenet_model_file_path = data["inputs"]["shapenet_model_file_path"]
        trans_matrix = np.array(data["inputs"]["trans_matrix"])[0]

        mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)

        bbox = mesh.get_axis_aligned_bounding_box()
        min_point = bbox.min_bound
        max_point = bbox.max_bound
        abb = np.hstack((min_point, max_point))
        obb_pcd = getABBPCD(abb)

        obb_pcd.paint_uniform_color(color_map[i % color_map.shape[0]] / 255.0)
        obb_pcd.transform(trans_matrix)
        obb_pcd.translate(translate)
        obb_list.append(obb_pcd)
    return obb_list


def getCOBOBBList(data_list, color_map=None, translate=[0, 0, 0]):
    translate = np.array(translate, dtype=float)

    obb_list = []

    if color_map is None:
        color_map = []
        for i in range(100):
            random_color = np.random.rand(3) * 255.0
            color_map.append(random_color)
    color_map = np.array(color_map, dtype=float)

    for i, data in enumerate(data_list):
        center = data["predictions"]["center"]
        rotate_matrix = data["predictions"]["rotate_matrix"]
        noc_translate = data["predictions"]["noc_translate"]
        noc_euler_angle = data["predictions"]["noc_euler_angle"]
        noc_scale = data["predictions"]["noc_scale"]
        noc_bbox = data["predictions"]["noc_bbox"]

        noc_obb = getOBBFromABB(noc_bbox)

        noc_translate_inv, noc_euler_angle_inv, noc_scale_inv = getInverseTrans(
            noc_translate, noc_euler_angle, noc_scale
        )

        noc_obb = transPointArray(
            noc_obb, noc_translate_inv, noc_euler_angle_inv, noc_scale_inv
        )
        noc_obb = noc_obb @ rotate_matrix
        noc_obb = noc_obb + center

        pcd = getOBBPCD(noc_obb, source_color)
        pcd.paint_uniform_color(color_map[i % color_map.shape[0]] / 255.0)
        pcd.translate(translate)
        obb_list.append(pcd)
    return obb_list


def getRefineOBBList(data_list, color_map=None, translate=[0, 0, 0]):
    translate = np.array(translate, dtype=float)

    obb_list = []

    if color_map is None:
        color_map = []
        for i in range(100):
            random_color = np.random.rand(3) * 255.0
            color_map.append(random_color)
    color_map = np.array(color_map, dtype=float)

    for i, data in enumerate(data_list):
        refine_noc_obb = data["predictions"]["refine_noc_obb"]
        refine_transform = data["predictions"]["refine_transform"]

        trans_noc_obb = transPoints(refine_noc_obb, refine_transform)
        pcd = getOBBPCD(trans_noc_obb)
        pcd.paint_uniform_color(color_map[i % color_map.shape[0]] / 255.0)
        pcd.translate(translate)
        obb_list.append(pcd)
    return obb_list
