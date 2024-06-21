#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

import numpy as np
import open3d as o3d
from copy import deepcopy

from noc_transform.Method.transform import transPoints
from points_shape_detect.Method.trans import normalizePointArray
from points_shape_detect.Method.matrix import getRotateMatrix


def renderRetrievalResult(obb_info_folder_path,
                          retrieval_cad_model_file_path_list):
    assert os.path.exists(obb_info_folder_path)

    object_folder_path = obb_info_folder_path + "object/"
    object_pcd_filename_list = os.listdir(object_folder_path)

    object_pcd_list = []
    obb_trans_matrix_list = []

    for object_pcd_filename in object_pcd_filename_list:
        if object_pcd_filename[-4:] != ".pcd":
            continue
        object_label = object_pcd_filename.split(".pcd")[0]

        object_file_path = obb_info_folder_path + "object/" + object_label + ".pcd"
        obb_trans_matrix_file_path = obb_info_folder_path + "obb_trans_matrix/" + object_label + ".json"
        assert os.path.exists(object_file_path)
        assert os.path.exists(obb_trans_matrix_file_path)

        object_pcd = o3d.io.read_point_cloud(object_file_path)
        object_pcd_list.append(object_pcd)

        with open(obb_trans_matrix_file_path, 'r') as f:
            obb_trans_matrix_dict = json.load(f)
        noc_trans_matrix = np.array(obb_trans_matrix_dict['noc_trans_matrix'])
        obb_trans_matrix_list.append(noc_trans_matrix)

    render_list = []

    row_num = np.sqrt(len(retrieval_cad_model_file_path_list))
    row_idx = 0
    col_idx = 0
    delta_trans = 1
    for i in range(len(retrieval_cad_model_file_path_list)):
        object_pcd = deepcopy(object_pcd_list[i])
        obb_trans_matrix = obb_trans_matrix_list[i]
        cad_model_file_path = retrieval_cad_model_file_path_list[i]
        cad_mesh = o3d.io.read_triangle_mesh(cad_model_file_path)

        points = np.array(object_pcd.points)
        points = transPoints(points, obb_trans_matrix)
        rotate_matrix = getRotateMatrix([90, 0, -90], True)
        points = points @ rotate_matrix
        object_pcd.points = o3d.utility.Vector3dVector(points)

        delta_translate = [row_idx * delta_trans, col_idx * delta_trans, 0]

        object_pcd.translate(delta_translate)
        cad_mesh.translate(delta_translate)

        cad_mesh.compute_triangle_normals()

        render_list.append(object_pcd)
        render_list.append(cad_mesh)

        row_idx += 1
        if row_idx == row_num:
            row_idx = 0
            col_idx += 1

    o3d.visualization.draw_geometries(render_list)
    return True


def renderRetrievalResult(obb_info_folder_path,
                          retrieval_cad_model_file_path_list,
                          layout_mesh_file_path_list,
                          need_transform=True):
    assert os.path.exists(obb_info_folder_path)

    object_folder_path = obb_info_folder_path + "object/"
    object_pcd_filename_list = os.listdir(object_folder_path)

    object_pcd_list = []
    obb_trans_matrix_list = []

    for object_pcd_filename in object_pcd_filename_list:
        if object_pcd_filename[-4:] != ".pcd":
            continue
        object_label = object_pcd_filename.split(".pcd")[0]

        object_file_path = obb_info_folder_path + "object/" + object_label + ".pcd"
        obb_trans_matrix_file_path = obb_info_folder_path + "obb_trans_matrix/" + object_label + ".json"
        assert os.path.exists(object_file_path)
        assert os.path.exists(obb_trans_matrix_file_path)

        object_pcd = o3d.io.read_point_cloud(object_file_path)
        object_pcd_list.append(object_pcd)

        with open(obb_trans_matrix_file_path, 'r') as f:
            obb_trans_matrix_dict = json.load(f)
        noc_trans_matrix = np.array(obb_trans_matrix_dict['noc_trans_matrix'])
        trans_matrix = np.linalg.inv(noc_trans_matrix)
        obb_trans_matrix_list.append(trans_matrix)

    if False:
        test_cad_model_file_path = "/home/chli/chLi/auto-scan2cad/1.ply"
        test_mesh = o3d.io.read_triangle_mesh(test_cad_model_file_path)

        points = np.array(test_mesh.vertices)
        points = normalizePointArray(points)
        test_mesh.vertices = o3d.utility.Vector3dVector(points)

        object_pcd = deepcopy(object_pcd_list[0])
        obb_trans_matrix = obb_trans_matrix_list[0]
        noc_trans_matrix = np.linalg.inv(obb_trans_matrix)

        points = np.array(object_pcd.points)
        points = transPoints(points, noc_trans_matrix)
        rotate_matrix = getRotateMatrix([90, 0, -90], True)
        points = points @ rotate_matrix
        object_pcd.points = o3d.utility.Vector3dVector(points)

        test_mesh.compute_triangle_normals()

        o3d.visualization.draw_geometries([object_pcd, test_mesh])
        exit()

    if False:
        object_pcd = deepcopy(object_pcd_list[0])
        obb_trans_matrix = obb_trans_matrix_list[0]

        test_cad_model_file_path = "/home/chli/chLi/auto-scan2cad/1.ply"
        test_mesh = o3d.io.read_triangle_mesh(test_cad_model_file_path)

        points = np.array(test_mesh.vertices)
        points = normalizePointArray(points)
        rotate_matrix = getRotateMatrix([-90, 0, 90], False)
        points = points @ rotate_matrix
        points = transPoints(points, obb_trans_matrix)
        test_mesh.vertices = o3d.utility.Vector3dVector(points)

        test_mesh.compute_triangle_normals()

        o3d.visualization.draw_geometries([object_pcd, test_mesh])
        exit()

    render_list = []

    for i in range(len(retrieval_cad_model_file_path_list)):
        object_pcd = object_pcd_list[i]
        obb_trans_matrix = obb_trans_matrix_list[i]
        cad_model_file_path = retrieval_cad_model_file_path_list[i]
        cad_mesh = o3d.io.read_triangle_mesh(cad_model_file_path)

        if need_transform:
            points = np.array(cad_mesh.vertices)
            points = normalizePointArray(points)
            rotate_matrix = getRotateMatrix([-90, 0, 90], False)
            points = points @ rotate_matrix
            points = transPoints(points, obb_trans_matrix)
            cad_mesh.vertices = o3d.utility.Vector3dVector(points)

        cad_mesh.compute_triangle_normals()

        render_list.append(object_pcd)
        render_list.append(cad_mesh)

    for layout_mesh_file_path in layout_mesh_file_path_list:
        layout_mesh = o3d.io.read_triangle_mesh(layout_mesh_file_path)
        layout_mesh.compute_triangle_normals()
        render_list.append(layout_mesh)

    o3d.visualization.draw_geometries(render_list)
    return True
