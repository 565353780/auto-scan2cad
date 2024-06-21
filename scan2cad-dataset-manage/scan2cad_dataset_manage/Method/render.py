#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d

from scan2cad_dataset_manage.Method.bbox import getOpen3DBBoxFromBBoxArray


def renderScan2CADScene(scene, scannet_scene_file_path=None):
    shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"

    mesh_list = []

    if scannet_scene_file_path is not None:
        scannet_scene_mesh = o3d.io.read_triangle_mesh(scannet_scene_file_path)
        mesh_list.append(scannet_scene_mesh)

    bbox_list = []

    for shapenet_model in scene.model_list:
        bbox_list.append(getOpen3DBBoxFromBBoxArray(shapenet_model.trans_bbox_array))
        shapenet_model_file_path = shapenet_dataset_folder_path + \
            shapenet_model.cad_cat_id + "/" + shapenet_model.cad_id + "/models/model_normalized.obj"
        assert os.path.exists(shapenet_model_file_path)
        shapenet_model_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
        shapenet_model_mesh.compute_vertex_normals()
        shapenet_model_mesh.transform(shapenet_model.trans_model_to_scan_matrix)
        mesh_list.append(shapenet_model_mesh)

    o3d.visualization.draw_geometries(bbox_list + mesh_list)
    return True

def renderScan2CADObjectModelMap(shapenet_model_dict, scannet_scene_file_path=None):
    scannet_object_file_path = shapenet_model_dict['scannet_object_file_path']
    assert os.path.exists(scannet_object_file_path)

    shapenet_model_file_path = shapenet_model_dict['shapenet_model_file_path']
    assert os.path.exists(shapenet_model_file_path)

    mesh_list = []

    if scannet_scene_file_path is not None:
        scannet_scene_mesh = o3d.io.read_triangle_mesh(scannet_scene_file_path)
        mesh_list.append(scannet_scene_mesh)

    scannet_object_mesh = o3d.io.read_triangle_mesh(scannet_object_file_path)
    mesh_list.append(scannet_object_mesh)

    shapenet_model_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
    shapenet_model_mesh.compute_vertex_normals()
    shapenet_model_trans_matrix = np.array(shapenet_model_dict['trans_matrix'])
    shapenet_model_mesh.transform(shapenet_model_trans_matrix)
    mesh_list.append(shapenet_model_mesh)

    bbox_list = []

    shapenet_model_trans_bbox_array = np.array(shapenet_model_dict['trans_bbox_array'])
    shapenet_model_trans_bbox = getOpen3DBBoxFromBBoxArray(shapenet_model_trans_bbox_array)
    bbox_list.append(shapenet_model_trans_bbox)

    o3d.visualization.draw_geometries(bbox_list + mesh_list)
    return True
