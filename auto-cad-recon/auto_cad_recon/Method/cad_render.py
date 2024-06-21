#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

from auto_cad_recon.Method.box_render import getABBPCD
from points_shape_detect.Method.trans import getInverseTrans, transPointArray


def getGTMeshInfoList(data_list, translate=[0, 0, 0]):
    translate = np.array(translate, dtype=float)

    mesh_info_list = []

    for data in data_list:
        shapenet_model_file_path = data["inputs"]["shapenet_model_file_path"]
        trans_matrix = np.array(data["inputs"]["trans_matrix"])[0]

        mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)

        bbox = mesh.get_axis_aligned_bounding_box()
        min_point = bbox.min_bound
        max_point = bbox.max_bound
        abb = np.hstack((min_point, max_point))
        obb_pcd = getABBPCD(abb)

        obb_pcd.transform(trans_matrix)
        obb_pcd.translate(translate)

        mesh_info_list.append([shapenet_model_file_path, trans_matrix, obb_pcd])
    return mesh_info_list


def getGTMeshList(data_list, color_map=None, translate=[0, 0, 0]):
    translate = np.array(translate, dtype=float)

    mesh_list = []

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
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color_map[i % color_map.shape[0]] / 255.0)
        mesh.transform(trans_matrix)
        mesh.translate(translate)
        mesh_list.append(mesh)
    return mesh_list


def getCOBMeshList(data_list, color_map=None, translate=[0, 0, 0]):
    translate = np.array(translate, dtype=float)

    mesh_list = []

    if color_map is None:
        color_map = []
        for i in range(100):
            random_color = np.random.rand(3) * 255.0
            color_map.append(random_color)
    color_map = np.array(color_map, dtype=float)

    for i, data in enumerate(data_list):
        shapenet_model_file_path = data["inputs"]["shapenet_model_file_path"]
        center = data["predictions"]["center"]
        rotate_matrix = data["predictions"]["rotate_matrix"]
        noc_translate = data["predictions"]["noc_translate"]
        noc_euler_angle = data["predictions"]["noc_euler_angle"]
        noc_scale = data["predictions"]["noc_scale"]

        noc_translate_inv, noc_euler_angle_inv, noc_scale_inv = getInverseTrans(
            noc_translate, noc_euler_angle, noc_scale
        )

        cob_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
        points = np.array(cob_mesh.vertices)

        points = transPointArray(
            points, noc_translate_inv, noc_euler_angle_inv, noc_scale_inv
        )
        points = points @ rotate_matrix
        points = points + center

        cob_mesh.vertices = o3d.utility.Vector3dVector(points)
        cob_mesh.compute_vertex_normals()
        cob_mesh.paint_uniform_color(color_map[i % color_map.shape[0]] / 255.0)
        cob_mesh.translate(translate)
        mesh_list.append(cob_mesh)
    return mesh_list


def getRefineMeshList(data_list, color_map=None, translate=[0, 0, 0]):
    translate = np.array(translate, dtype=float)

    mesh_list = []

    if color_map is None:
        color_map = []
        for i in range(100):
            random_color = np.random.rand(3) * 255.0
            color_map.append(random_color)
    color_map = np.array(color_map, dtype=float)

    for i, data in enumerate(data_list):
        shapenet_model_file_path = data["inputs"]["shapenet_model_file_path"]
        refine_transform = data["predictions"]["refine_transform"]

        refine_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
        points = np.array(refine_mesh.vertices)
        points = normalizePointArray(points)
        points = transPoints(points, refine_transform)

        refine_mesh.vertices = o3d.utility.Vector3dVector(points)
        refine_mesh.compute_vertex_normals()
        refine_mesh.paint_uniform_color(color_map[i % color_map.shape[0]] / 255.0)
        refine_mesh.translate(translate)
        mesh_list.append(refine_mesh)
    return mesh_list


def getRetrievalMeshList(data_list, color_map=None, translate=[0, 0, 0]):
    translate = np.array(translate, dtype=float)

    mesh_list = []

    if color_map is None:
        color_map = []
        for i in range(100):
            random_color = np.random.rand(3) * 255.0
            color_map.append(random_color)
    color_map = np.array(color_map, dtype=float)

    for i, data in enumerate(data_list):
        shapenet_model_file_path = data["predictions"]["retrieval_model_file_path"]
        trans_matrix = np.array(data["inputs"]["trans_matrix"])[0]

        mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color_map[i % color_map.shape[0]] / 255.0)
        mesh.transform(trans_matrix)
        mesh.translate(translate)
        mesh_list.append(mesh)
    return mesh_list
