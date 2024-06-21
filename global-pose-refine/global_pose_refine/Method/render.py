#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from auto_cad_recon.Method.bbox import getOBBFromABB
from points_shape_detect.Data.bbox import BBox
from points_shape_detect.Method.bbox import (getOpen3DBBox,
                                             getOpen3DBBoxFromBBox)
from scene_layout_detect.Method.mesh import generateLayoutMesh


def getPCDFromPointArray(point_array, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)

    if color is not None:
        colors = np.array([color for _ in range(point_array.shape[0])],
                          dtype=float) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def getOBBPCD(point_array, color=None):
    lines = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(point_array),
        lines=o3d.utility.Vector2iVector(lines))

    if color is not None:
        colors = np.array([color
                           for i in range(len(lines))], dtype=float) / 255.0
        line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def renderPointArray(point_array):
    if isinstance(point_array, np.ndarray):
        pcd = getPCDFromPointArray(point_array)
    else:
        pcd = getPCDFromPointArray(point_array.detach().cpu().numpy())

    o3d.visualization.draw_geometries([pcd])
    return True


def renderPointArrayList(point_array_list):
    if isinstance(point_array_list[0], np.ndarray):
        points = np.vstack(point_array_list)
        return renderPointArray(points)

    points = torch.vstack(point_array_list)
    return renderPointArray(points)


def renderRefineBBox(data):
    assert 'floor_position' in data['inputs'].keys()
    assert 'object_obb' in data['inputs'].keys()
    assert 'object_abb' in data['inputs'].keys()
    assert 'object_obb_center' in data['inputs'].keys()
    assert 'trans_object_obb' in data['inputs'].keys()
    assert 'trans_object_abb' in data['inputs'].keys()
    assert 'trans_object_obb_center' in data['inputs'].keys()

    render_list = []

    floor_position = data['inputs']['floor_position'][0].cpu().numpy().reshape(
        -1, 3)
    layout_mesh = generateLayoutMesh(floor_position)
    render_list.append(layout_mesh)

    gt_obb_list = data['inputs']['object_obb'][0].cpu().numpy().reshape(
        -1, 8, 3)
    for gt_obb in gt_obb_list:
        pcd = getOBBPCD(gt_obb, [0, 255, 0])
        render_list.append(pcd)

    gt_abb_list = data['inputs']['object_abb'][0].cpu().numpy().reshape(-1, 6)
    for gt_abb in gt_abb_list:
        obb = getOBBFromABB(gt_abb)
        pcd = getOBBPCD(obb, [0, 255, 0])
        pcd.translate([0, 0, 3])
        render_list.append(pcd)

    gt_obb_center_list = data['inputs']['object_obb_center'][0].cpu().numpy(
    ).reshape(-1, 1, 3)
    for gt_obb_center in gt_obb_center_list:
        pcd = getPCDFromPointArray(gt_obb_center, [0, 255, 0])
        render_list.append(pcd)

    trans_obb_list = data['inputs']['trans_object_obb'][0].cpu().numpy(
    ).reshape(-1, 8, 3)
    for trans_obb in trans_obb_list:
        pcd = getOBBPCD(trans_obb, [0, 0, 255])
        render_list.append(pcd)

    trans_abb_list = data['inputs']['trans_object_abb'][0].cpu().numpy(
    ).reshape(-1, 6)
    for trans_abb in trans_abb_list:
        obb = getOBBFromABB(trans_abb)
        pcd = getOBBPCD(obb, [0, 0, 255])
        pcd.translate([0, 0, 3])
        render_list.append(pcd)

    trans_obb_center_list = data['inputs']['trans_object_obb_center'][0].cpu(
    ).numpy().reshape(-1, 1, 3)
    for trans_obb_center in trans_obb_center_list:
        pcd = getPCDFromPointArray(trans_obb_center, [0, 0, 255])
        render_list.append(pcd)

    if 'refine_object_obb' in data['predictions'].keys():
        refine_obb_list = data['predictions']['refine_object_obb'][0].detach(
        ).cpu().numpy().reshape(-1, 8, 3)
        for refine_obb in refine_obb_list:
            pcd = getOBBPCD(refine_obb, [255, 0, 0])
            render_list.append(pcd)

    if 'refine_object_abb' in data['predictions'].keys():
        refine_abb_list = data['predictions']['refine_object_abb'][0].detach(
        ).cpu().numpy().reshape(-1, 6)
        for refine_abb in refine_abb_list:
            obb = getOBBFromABB(refine_abb)
            pcd = getOBBPCD(obb, [255, 0, 0])
            pcd.translate([0, 0, 3])
            render_list.append(pcd)

    if 'refine_object_obb_center' in data['predictions'].keys():
        refine_obb_center_list = data['predictions'][
            'refine_object_obb_center'][0].detach().cpu().numpy().reshape(
                -1, 1, 3)
        for refine_obb_center in refine_obb_center_list:
            pcd = getPCDFromPointArray(refine_obb_center, [255, 0, 0])
            render_list.append(pcd)

    o3d.visualization.draw_geometries(render_list)
    return True
