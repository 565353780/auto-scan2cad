#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from copy import deepcopy
from threading import Thread

from points_shape_detect.Method.trans import (getInverseTrans,
                                              normalizePointArray,
                                              transPointArray)
from scan2cad_dataset_manage.Method.matrix import make_M_from_tqs

from auto_cad_recon.Config.color import COLOR_MAP
from auto_cad_recon.Method.bbox import getOBBFromABB
from auto_cad_recon.Method.box_render import (getABBPCD, getCOBOBBList,
                                              getGTOBBList, getOBBPCD,
                                              getRefineOBBList)
from auto_cad_recon.Method.cad_render import (getCOBMeshList, getGTMeshList,
                                              getRefineMeshList,
                                              getRetrievalMeshList)
from auto_cad_recon.Method.match_check import isMatch
from auto_cad_recon.Method.object_render import getObjectPCDList
from auto_cad_recon.Method.scene_render import getScenePCDList
from auto_cad_recon.Method.trans import transPoints
from auto_cad_recon.Method.scene_render import getMergedScenePCD


def renderMergedScene(point_image_list,
                      background_only=False,
                      is_gray=False,
                      translate=[0, 0, 0],
                      estimate_normals=False):
    merged_scene_pcd = getMergedScenePCD(point_image_list, background_only,
                                         is_gray, translate, estimate_normals)
    drawGeometries([merged_scene_pcd], "Merged Scene PCD")
    return True


def getLayoutMesh(layout_data):
    layout_mesh = layout_data['predictions']['layout_mesh']
    return layout_mesh


def getRetrieval(data_list, color=[255, 0, 0], translate=[0, 0, 0]):
    mesh_list = []

    source_color = deepcopy(color)
    color = np.array(color, dtype=float) / 255.0
    translate = np.array(translate, dtype=float)

    for data in data_list:
        refine_transform = data['predictions']['refine_transform']
        retrieval_model_file_path = data['predictions'][
            'retrieval_model_file_path']

        mesh = o3d.io.read_triangle_mesh(retrieval_model_file_path)
        points = np.array(mesh.vertices)
        points = normalizePointArray(points)
        points = transPoints(points, refine_transform)
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        mesh.translate(translate)
        mesh_list.append(mesh)
    return mesh_list


def getRenderGeometries(data_list):
    gt_mesh_list = []
    mesh_list = []

    for data in data_list:
        object_file_name = data['inputs']['object_label'].split("==object")[0]
        shapenet_model_file_path = data['inputs']['shapenet_model_file_path']
        trans_matrix = np.array(data['inputs']['trans_matrix'])[0]
        retrieval_model_file_path = data['predictions'][
            'retrieval_model_file_path']
        retrieval_object_file_name = data['predictions'][
            'retrieval_object_file_name']

        mesh = o3d.io.read_triangle_mesh(retrieval_model_file_path)
        mesh.compute_vertex_normals()
        if isMatch(object_file_name, retrieval_object_file_name):
            mesh.paint_uniform_color([0, 1, 0])
        else:
            mesh.paint_uniform_color([1, 0, 0])
            gt_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
            gt_mesh.compute_vertex_normals()
            gt_mesh.paint_uniform_color([0, 0, 1])
            gt_mesh.transform(trans_matrix)
            gt_mesh_list.append(gt_mesh)

        mesh.transform(trans_matrix)
        mesh_list.append(mesh)
    return gt_mesh_list + mesh_list


def getOriginResult(data_list, translate=[0.0, 0.0, 0.0]):
    gt_mesh_list = []
    mesh_list = []

    delta_x = np.array([2.0, 0.0, 0.0], dtype=float)
    delta_y = np.array([0.0, 2.0, 0.0], dtype=float)
    object_num = len(data_list)
    row_num = int(np.sqrt(object_num))

    for i, data in enumerate(data_list):
        row_idx = int(i / row_num)
        col_idx = i - row_num * row_idx

        delta_translate = delta_x * row_idx + delta_y * col_idx + translate

        merged_point_array = data['predictions']['merged_point_array']
        shapenet_model_file_path = data['inputs']['shapenet_model_file_path']
        trans_matrix = data['inputs']['trans_matrix'][0]
        center = data['predictions']['center']

        origin_points = merged_point_array - center

        mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
        mesh.transform(trans_matrix)
        points = np.array(mesh.vertices)
        points = points - center
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0, 1, 0])
        mesh.translate(delta_translate)
        gt_mesh_list.append(mesh)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(origin_points)
        colors = np.zeros_like(origin_points, dtype=float)
        colors[:] = np.array([255, 0, 0], dtype=float) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.translate(delta_translate)
        mesh_list.append(pcd)
    return gt_mesh_list + mesh_list


def getRotateBackResult(data_list, translate=[0.0, 0.0, 0.0]):
    gt_mesh_list = []
    mesh_list = []

    delta_x = np.array([2.0, 0.0, 0.0], dtype=float)
    delta_y = np.array([0.0, 2.0, 0.0], dtype=float)
    object_num = len(data_list)
    row_num = int(np.sqrt(object_num))

    for i, data in enumerate(data_list):
        row_idx = int(i / row_num)
        col_idx = i - row_num * row_idx

        delta_translate = delta_x * row_idx + delta_y * col_idx + translate

        merged_point_array = data['predictions']['merged_point_array']
        shapenet_model_file_path = data['inputs']['shapenet_model_file_path']
        trans_matrix = data['inputs']['trans_matrix'][0]
        center = data['predictions']['center']
        rotate_matrix_inv = data['predictions']['rotate_matrix_inv']

        origin_points = merged_point_array - center

        rotate_back_points = origin_points @ rotate_matrix_inv

        mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
        mesh.transform(trans_matrix)
        points = np.array(mesh.vertices)
        points = points - center
        points = points @ rotate_matrix_inv
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0, 1, 0])
        mesh.translate(delta_translate)
        gt_mesh_list.append(mesh)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rotate_back_points)
        colors = np.zeros_like(rotate_back_points, dtype=float)
        colors[:] = np.array([255, 0, 0], dtype=float) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.translate(delta_translate)
        mesh_list.append(pcd)
    return gt_mesh_list + mesh_list


def getCABResult(data_list, translate=[0.0, 0.0, 0.0]):
    gt_mesh_list = []
    mesh_list = []
    gt_obb_list = []
    obb_list = []

    delta_x = np.array([2.0, 0.0, 0.0], dtype=float)
    delta_y = np.array([0.0, 2.0, 0.0], dtype=float)
    object_num = len(data_list)
    row_num = int(np.sqrt(object_num))

    for i, data in enumerate(data_list):
        row_idx = int(i / row_num)
        col_idx = i - row_num * row_idx

        delta_translate = delta_x * row_idx + delta_y * col_idx + translate

        merged_point_array = data['predictions']['merged_point_array']
        shapenet_model_file_path = data['inputs']['shapenet_model_file_path']
        center = data['predictions']['center']
        rotate_matrix_inv = data['predictions']['rotate_matrix_inv']
        trans_matrix = data['inputs']['trans_matrix'][0]
        origin_bbox = data['predictions']['origin_bbox']

        mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
        mesh.transform(trans_matrix)
        points = np.array(mesh.vertices)
        points = points - center
        points = points @ rotate_matrix_inv
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0, 1, 0])
        mesh.translate(delta_translate)
        gt_mesh_list.append(mesh)

        origin_points = merged_point_array - center

        rotate_back_points = origin_points @ rotate_matrix_inv

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(rotate_back_points)
        colors = np.zeros_like(rotate_back_points, dtype=float)
        colors[:] = np.array([255, 0, 0], dtype=float) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.translate(delta_translate)
        mesh_list.append(pcd)

        pcd = getABBPCD(origin_bbox, [255, 0, 0])
        pcd.translate(delta_translate)
        obb_list.append(pcd)

        cad_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
        cad_mesh.transform(trans_matrix)
        points = np.array(cad_mesh.vertices)
        points = points - center
        points = points @ rotate_matrix_inv
        cad_mesh.vertices = o3d.utility.Vector3dVector(points)
        cad_bbox = cad_mesh.get_axis_aligned_bounding_box()
        min_point = cad_bbox.min_bound
        max_point = cad_bbox.max_bound
        cad_abb = np.hstack((min_point, max_point))
        gt_obb_pcd = getABBPCD(cad_abb, [0, 255, 0])
        gt_obb_pcd.translate(delta_translate)
        gt_obb_list.append(gt_obb_pcd)
    return gt_mesh_list + mesh_list + gt_obb_list + obb_list


def getNOCResult(data_list, translate=[0.0, 0.0, 0.0]):
    gt_mesh_list = []
    mesh_list = []
    gt_obb_list = []
    obb_list = []

    delta_x = np.array([2.0, 0.0, 0.0], dtype=float)
    delta_y = np.array([0.0, 2.0, 0.0], dtype=float)
    object_num = len(data_list)
    row_num = int(np.sqrt(object_num))

    for i, data in enumerate(data_list):
        row_idx = int(i / row_num)
        col_idx = i - row_num * row_idx

        delta_translate = delta_x * row_idx + delta_y * col_idx + translate

        merged_point_array = data['predictions']['merged_point_array']
        shapenet_model_file_path = data['inputs']['shapenet_model_file_path']
        center = data['predictions']['center']
        rotate_matrix_inv = data['predictions']['rotate_matrix_inv']
        trans_matrix = data['inputs']['trans_matrix'][0]
        noc_translate = data['predictions']['noc_translate']
        noc_euler_angle = data['predictions']['noc_euler_angle']
        noc_scale = data['predictions']['noc_scale']
        noc_bbox = data['predictions']['noc_bbox']

        mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
        mesh.transform(trans_matrix)
        points = np.array(mesh.vertices)
        points = points - center
        points = points @ rotate_matrix_inv
        points = transPointArray(points,
                                 noc_translate,
                                 noc_euler_angle,
                                 noc_scale,
                                 is_inverse=True)
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0, 1, 0])
        mesh.translate(delta_translate)
        gt_mesh_list.append(mesh)

        origin_points = merged_point_array - center
        rotate_back_points = origin_points @ rotate_matrix_inv
        noc_points = transPointArray(rotate_back_points,
                                     noc_translate,
                                     noc_euler_angle,
                                     noc_scale,
                                     is_inverse=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(noc_points)
        pcd.paint_uniform_color([1, 0, 0])
        pcd.translate(delta_translate)
        mesh_list.append(pcd)

        pcd = getABBPCD(noc_bbox, [255, 0, 0])
        pcd.translate(delta_translate)
        obb_list.append(pcd)

        cad_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
        cad_mesh.transform(trans_matrix)
        points = np.array(cad_mesh.vertices)
        points = points - center
        points = points @ rotate_matrix_inv
        points = transPointArray(points,
                                 noc_translate,
                                 noc_euler_angle,
                                 noc_scale,
                                 is_inverse=True)
        cad_mesh.vertices = o3d.utility.Vector3dVector(points)
        cad_bbox = cad_mesh.get_axis_aligned_bounding_box()
        min_point = cad_bbox.min_bound
        max_point = cad_bbox.max_bound
        cad_abb = np.hstack((min_point, max_point))
        gt_obb_pcd = getABBPCD(cad_abb, [0, 255, 0])
        gt_obb_pcd.translate(delta_translate)
        gt_obb_list.append(gt_obb_pcd)
    return gt_mesh_list + mesh_list + gt_obb_list + obb_list


def drawGeometries(mesh_list, window_name="Open3D"):
    thread = Thread(target=o3d.visualization.draw_geometries,
                    args=(mesh_list, window_name))
    thread.start()
    return True


def renderOriginResult(data_list):
    mesh_list = getOriginResult(data_list)
    drawGeometries(mesh_list, "Origin")
    return True


def renderRotateBackResult(data_list):
    mesh_list = getRotateBackResult(data_list)
    drawGeometries(mesh_list, "RotateBack")
    return True


def renderCABResult(data_list):
    mesh_list = getCABResult(data_list)
    drawGeometries(mesh_list, "CAB")
    return True


def renderNOCResult(data_list):
    mesh_list = getNOCResult(data_list)
    drawGeometries(mesh_list, "NOC")
    return True


def getObjectRenderList(data_list=None,
                        object_mode='none',
                        obb_mode='none',
                        mesh_mode='none',
                        object_color_map=COLOR_MAP,
                        object_translate=[0, 0, 0],
                        obb_color_map=COLOR_MAP,
                        obb_translate=[0, 0, 0],
                        mesh_color_map=COLOR_MAP,
                        mesh_translate=[0, 0, 0]):
    '''
    object_mode: ['none', 'scan']
    obb_mode: ['none', 'cob', 'refine', 'gt']
    mesh_mode: ['none', 'cob', 'refine', 'gt']
    '''

    render_list = []
    window_name = ""
    label_list = []

    if object_mode == 'scan':
        object_pcd_list = getObjectPCDList(data_list, object_color_map,
                                           object_translate, False)
        render_list += object_pcd_list
        window_name += '[Object-Scan]'
        for i in range(len(object_pcd_list)):
            label_list.append('object_pcd_' + str(i) + '.pcd')

    if obb_mode == 'cob':
        obb_list = getCOBOBBList(data_list, obb_color_map, obb_translate)
        render_list += obb_list
        window_name += '[COB-OBB]'
        for i in range(len(obb_list)):
            label_list.append('object_obb_' + str(i) + '.lines')
    elif obb_mode == 'refine':
        obb_list = getRefineOBBList(data_list, obb_color_map, obb_translate)
        render_list += obb_list
        window_name += '[Refine-OBB]'
        for i in range(len(obb_list)):
            label_list.append('object_obb_' + str(i) + '.lines')
    elif obb_mode == 'gt':
        obb_list = getGTOBBList(data_list, obb_color_map, obb_translate)
        render_list += obb_list
        window_name += '[GT-OBB]'
        for i in range(len(obb_list)):
            label_list.append('object_obb_' + str(i) + '.lines')

    if mesh_mode == 'cob':
        mesh_list = getCOBMeshList(data_list, mesh_color_map, mesh_translate)
        render_list += mesh_list
        window_name += '[COB-Mesh]'
        for i in range(len(mesh_list)):
            label_list.append('cad_mesh_' + str(i) + '.ply')
    elif mesh_mode == 'refine':
        mesh_list = getRefineMeshList(data_list, mesh_color_map,
                                      mesh_translate)
        render_list += mesh_list
        window_name += '[Refine-Mesh]'
        for i in range(len(mesh_list)):
            label_list.append('cad_mesh_' + str(i) + '.ply')
    elif mesh_mode == 'gt':
        mesh_list = getGTMeshList(data_list, mesh_color_map, mesh_translate)
        render_list += mesh_list
        window_name += '[GT-Mesh]'
        for i in range(len(mesh_list)):
            label_list.append('cad_mesh_' + str(i) + '.ply')
    elif mesh_mode == 'retrieval':
        mesh_list = getRetrievalMeshList(data_list, mesh_color_map,
                                         mesh_translate)
        render_list += mesh_list
        window_name += '[Retrieval-Mesh]'
        for i in range(len(mesh_list)):
            label_list.append('cad_mesh_' + str(i) + '.ply')
    return render_list, window_name, label_list


def getSceneRenderList(layout_data=None,
                       point_image_list=None,
                       scene_background_only=True,
                       is_scene_gray=True,
                       scene_translate=[0, 0, 0],
                       estimate_scene_normals=False):
    render_list = []
    window_name = ""
    label_list = []

    if layout_data is not None:
        layout_mesh = getLayoutMesh(layout_data)
        render_list.append(layout_mesh)
        window_name += "[Layout]"
        label_list.append('layout.ply')

    if point_image_list is not None:
        scene_pcd_list = getScenePCDList(point_image_list,
                                         scene_background_only, is_scene_gray,
                                         scene_translate,
                                         estimate_scene_normals)
        render_list += scene_pcd_list
        window_name += "[Scene]"
        for i in range(len(scene_pcd_list)):
            label_list.append('scene_viewpoint_' + str(i) + '.pcd')
    return render_list, window_name, label_list


def renderResult(data_list=None,
                 layout_data=None,
                 point_image_list=None,
                 object_mode='none',
                 obb_mode='none',
                 mesh_mode='none',
                 object_color_map=COLOR_MAP,
                 object_translate=[0, 0, 0],
                 obb_color_map=COLOR_MAP,
                 obb_translate=[0, 0, 0],
                 mesh_color_map=COLOR_MAP,
                 mesh_translate=[0, 0, 0],
                 scene_background_only=True,
                 is_scene_gray=True,
                 scene_translate=[0, 0, 0],
                 estimate_scene_normals=False):
    '''
    object_mode: ['none', 'scan']
    obb_mode: ['none', 'cob', 'refine', 'gt']
    mesh_mode: ['none', 'cob', 'refine', 'gt']
    '''

    object_render_list, object_window_name, _ = getObjectRenderList(
        data_list, object_mode, obb_mode, mesh_mode, object_color_map,
        object_translate, obb_color_map, obb_translate, mesh_color_map,
        mesh_translate)

    scene_render_list, scene_window_name, _ = getSceneRenderList(
        layout_data, point_image_list, scene_background_only, is_scene_gray,
        scene_translate, estimate_scene_normals)

    render_list = object_render_list + scene_render_list
    window_name = object_window_name + scene_window_name

    drawGeometries(render_list, window_name)
    return True


def renderCOBResult(data_list, layout_data=None, scene_points_array=None):
    gt_mesh_list = getGTMesh(data_list, [0, 255, 0], [0, 0, 0])
    gt_obb_list = getGTOBB(data_list, [0, 255, 0], [0, 0, 0])
    cob_mesh_list = getCOBMesh(data_list, [255, 0, 0], [0, 0, 0])
    cob_obb_list = getCOBOBB(data_list, [255, 0, 0], [0, 0, 0])

    mesh_list = gt_mesh_list + gt_obb_list + cob_mesh_list + cob_obb_list

    renderGeometries(mesh_list, "COB", layout_data, scene_points_array, None)
    return True


def renderRefineResult(data_list, layout_data=None, scene_points_array=None):
    gt_mesh_list = getGTMesh(data_list, [0, 255, 0], [0, 0, 0])
    gt_obb_list = getGTOBB(data_list, [0, 255, 0], [0, 0, 0])
    refine_mesh_list = getRefineMesh(data_list, [255, 0, 0], [0, 0, 0])
    refine_obb_list = getRefineOBB(data_list, [255, 0, 0], [0, 0, 0])

    mesh_list = gt_mesh_list + gt_obb_list + refine_mesh_list + refine_obb_list

    renderGeometries(mesh_list, "Refine", layout_data, scene_points_array,
                     None)
    return True


def renderCOBAndRefineResult(data_list,
                             layout_data=None,
                             scene_points_array=None):
    gt_mesh_list = getGTMesh(data_list, [0, 255, 0], [0, 0, 0])
    gt_obb_list = getGTOBB(data_list, [0, 255, 0], [0, 0, 0])
    cob_mesh_list = getCOBMesh(data_list, [0, 0, 255], [0, 0, 0])
    cob_obb_list = getCOBOBB(data_list, [0, 0, 255], [0, 0, 0])
    refine_mesh_list = getRefineMesh(data_list, [255, 0, 0], [0, 0, 0])
    refine_obb_list = getRefineOBB(data_list, [255, 0, 0], [0, 0, 0])

    mesh_list = \
        gt_mesh_list + gt_obb_list + \
        cob_mesh_list + cob_obb_list + \
        refine_mesh_list + refine_obb_list

    renderGeometries(mesh_list, "COBAndRefine", layout_data,
                     scene_points_array, None)
    return True


def renderRetrievalResult(data_list,
                          layout_data=None,
                          scene_points_array=None):
    gt_mesh_list = getGTMesh(data_list, [0, 255, 0], [0, 0, 0])
    gt_obb_list = getGTOBB(data_list, [0, 255, 0], [0, 0, 0])
    retrieval_list = getRetrieval(data_list, [255, 0, 0], [0, 0, 0])

    mesh_list = gt_mesh_list + gt_obb_list + retrieval_list

    renderGeometries(mesh_list, "Retrieval", layout_data, scene_points_array,
                     None)
    return True


def renderPipeline(data_list, layout_data, point_image_list):
    # Initial Input
    # Scan Result
    renderResult(None,
                 None,
                 point_image_list,
                 'none',
                 'none',
                 'none',
                 scene_background_only=False,
                 is_scene_gray=False)

    # Instance Segmentation
    # Scan Result, Colored Object
    renderResult(data_list,
                 None,
                 point_image_list,
                 'scan',
                 'none',
                 'none',
                 is_scene_gray=True)

    # Refined COB
    # Colored Object, Refine COB
    renderResult(data_list, None, None, 'scan', 'refine', 'none')
    #  obb_color_map=[[255, 0, 0]])

    # GT COB
    # Layout, Colored Object, GT COB
    renderResult(data_list, layout_data, None, 'scan', 'gt', 'none')
    #  obb_color_map=[[0, 255, 0]])

    # Retrieval Objects
    # Colored Object, GT COB, GT Mesh
    renderResult(data_list,
                 None,
                 None,
                 'scan',
                 'gt',
                 'gt',
                 mesh_color_map=[[148, 148, 148]])
    #  obb_color_map=[[0, 255, 0]],

    # Retrieval Scene
    # Layout, GT Mesh
    renderResult(data_list,
                 layout_data,
                 None,
                 'none',
                 'none',
                 'gt',
                 mesh_color_map=[[148, 148, 148]])

    # Retrieval Scene With PointCloud
    # Layout, Retrieval Mesh
    renderResult(data_list,
                 layout_data,
                 point_image_list,
                 'scan',
                 'none',
                 'retrieval',
                 is_scene_gray=False)

    # GT Scene With PointCloud
    # Layout, GT Mesh
    renderResult(data_list,
                 layout_data,
                 point_image_list,
                 'scan',
                 'none',
                 'gt',
                 is_scene_gray=False)
    return True


def renderDataList(data_list, layout_data=None, point_image_list=None):
    return renderPipeline(data_list, layout_data, point_image_list)

    renderOriginResult(data_list)
    renderRotateBackResult(data_list)
    renderCABResult(data_list)
    renderNOCResult(data_list)

    renderCOBResult(data_list, layout_data, scene_points_array)
    renderRefineResult(data_list, layout_data, scene_points_array)
    renderCOBAndRefineResult(data_list, layout_data, scene_points_array)
    renderRetrievalResult(data_list, layout_data, scene_points_array)
    return True
