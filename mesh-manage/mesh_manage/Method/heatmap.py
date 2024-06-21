#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from tqdm import tqdm

from mesh_manage.Config.color import COLOR_MAP_DICT

from mesh_manage.Method.path import createFileFolder

def getICPTrans(source_pointcloud, target_pointcloud, move_list=None):
    threshold = 1.0
    trans_init = np.asarray([[1,0,0,0],
                             [0,1,0,0],
                             [0,0,1,0],
                             [0,0,0,1]], dtype=float)

    if move_list is not None:
        trans_init[:3, 3] = move_list

    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pointcloud, target_pointcloud, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p

def getNoise(point_num, noise_sigma, print_progress=False):
    noise_x = np.random.normal(0, noise_sigma, point_num)
    noise_y = np.random.normal(0, noise_sigma, point_num)
    noise_z = np.random.normal(0, noise_sigma, point_num)
    noise = []

    for_data = range(point_num)
    if print_progress:
        print("[INFO][heatmap::getNoise]")
        print("\t start generate noise...")
        for_data = tqdm(for_data)
    for i in for_data:
        noise.append([noise_x[i], noise_y[i], noise_z[i]])
    noise = np.array(noise, dtype=float)
    return noise

def getDistColorMap(dist_list, color_map, error_max=None, print_progress=False):
    colors = []
    color_num = len(color_map)
    min_dist = 0
    max_dist = error_max
    if max_dist is None:
        max_dist = np.max(dist_list)
    dist_step = (max_dist - min_dist) / (color_num - 1.0)

    for_data = dist_list
    if print_progress:
        print("[INFO][heatmap::getDistColorMap]")
        print("\t start generate heatmap...")
        for_data = tqdm(for_data)
    for dist in for_data:
        dist_divide = dist / dist_step
        color_idx = int(dist_divide)
        if color_idx >= color_num - 1:
            colors.append(color_map[color_num - 1])
            continue

        next_color_weight = dist_divide - color_idx
        color = (1.0 - next_color_weight) * color_map[color_idx]
        if next_color_weight > 0:
            color += next_color_weight * color_map[color_idx + 1]
        colors.append(color)

    colors = np.array(colors, dtype=float) / 255.0
    return colors

def getSpherePointCloud(pointcloud,
                        radius=1.0,
                        resolution=20,
                        print_progress=False):
    sphere_pointcloud = o3d.geometry.PointCloud()
    sphere_points_list = []
    sphere_colors = []
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius,
        resolution=resolution)
    sphere_points = np.array(mesh_sphere.vertices)

    points = np.array(pointcloud.points)
    colors = np.array(pointcloud.colors)
    for_data = range(len(points))
    if print_progress:
        print("[INFO][heatmap::getSpherePointCloud]")
        print("\t start generate sphere pointcloud...")
        for_data = tqdm(for_data)
    for i in for_data:
        new_points = sphere_points + points[i]
        sphere_points_list.append(new_points)
        for _ in sphere_points:
            sphere_colors.append(colors[i])
    points = np.concatenate(sphere_points_list, axis=0)
    colors = np.array(sphere_colors)
    sphere_pointcloud.points = \
        o3d.utility.Vector3dVector(points)
    sphere_pointcloud.colors = \
        o3d.utility.Vector3dVector(colors)
    return sphere_pointcloud

def getHeatMap(partial_mesh_file_path,
               complete_mesh_file_path,
               save_complete_mesh_file_path,
               color_map=COLOR_MAP_DICT["gray_blue"],
               move_list=None,
               partial_noise_sigma=0,
               error_max=None,
               use_icp=True,
               is_visual=False,
               print_progress=False):
    if print_progress:
        print("[INFO][heatmap::getHeatMap]")
        print("\t start load complete mesh...")
    complete_mesh = o3d.io.read_triangle_mesh(complete_mesh_file_path, print_progress=print_progress)
    complete_pointcloud = o3d.io.read_point_cloud(complete_mesh_file_path, print_progress=print_progress)

    if print_progress:
        print("[INFO][heatmap::getHeatMap]")
        print("\t start load partial pointcloud...")
    partial_pointcloud = o3d.io.read_point_cloud(partial_mesh_file_path, print_progress=print_progress)

    # FIXME: for fast forward company ply file only
    #  R = partial_pointcloud.get_rotation_matrix_from_xyz((-np.pi / 2.0, 0, 0))
    #  partial_pointcloud.rotate(R, center=(0, 0, 0))

    if use_icp:
        reg_p2p = getICPTrans(partial_pointcloud, complete_pointcloud, move_list)
        partial_pointcloud.transform(reg_p2p.transformation)

    if partial_noise_sigma > 0:
        partial_points = np.array(partial_pointcloud.points)
        noise = getNoise(partial_points.shape[0], partial_noise_sigma, print_progress)
        partial_points += noise
        partial_pointcloud.points = o3d.utility.Vector3dVector(partial_points)

    dist_to_partial = np.array(complete_pointcloud.compute_point_cloud_distance(partial_pointcloud))
    colors = getDistColorMap(dist_to_partial, color_map, error_max, print_progress)

    complete_pointcloud.colors = o3d.utility.Vector3dVector(colors)
    complete_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    complete_pointcloud.normals = o3d.utility.Vector3dVector()

    complete_mesh.compute_vertex_normals()

    #  sphere_complete_pointcloud = getSpherePointCloud(complete_pointcloud,
                                                     #  0.001,
                                                     #  20,
                                                     #  print_progress)

    if is_visual:
        print("[INFO][heatmap::getHeatMap]")
        print("\t start show heatmap result...")
        o3d.visualization.draw_geometries([partial_pointcloud, complete_mesh])

    createFileFolder(save_complete_mesh_file_path)

    if print_progress:
        print("[INFO][heatmap::getHeatMap]")
        print("\t start save heatmap...")
    o3d.io.write_triangle_mesh(save_complete_mesh_file_path, complete_mesh,
                               write_ascii=True, print_progress=print_progress)
    return True

def getDiffHeatMap(partial_mesh_1_file_path,
                   partial_mesh_2_file_path,
                   complete_mesh_file_path,
                   save_complete_mesh_file_path,
                   color_map=COLOR_MAP_DICT["jet"],
                   move_list=None,
                   partial_noise_sigma=0,
                   error_max=None,
                   is_visual=False,
                   print_progress=False):
    if print_progress:
        print("[INFO][heatmap::getDiffHeatMap]")
        print("\t start load complete mesh...")
    complete_mesh = o3d.io.read_triangle_mesh(complete_mesh_file_path, print_progress=print_progress)
    complete_pointcloud = o3d.io.read_point_cloud(complete_mesh_file_path, print_progress=print_progress)

    if print_progress:
        print("[INFO][heatmap::getDiffHeatMap]")
        print("\t start load partial pointcloud...")
    partial_pointcloud_1 = o3d.io.read_point_cloud(partial_mesh_1_file_path, print_progress=print_progress)
    partial_pointcloud_2 = o3d.io.read_point_cloud(partial_mesh_2_file_path, print_progress=print_progress)

    reg_p2p = getICPTrans(partial_pointcloud_1, complete_pointcloud, move_list)
    partial_pointcloud_1.transform(reg_p2p.transformation)

    reg_p2p = getICPTrans(partial_pointcloud_2, complete_pointcloud, move_list)
    partial_pointcloud_2.transform(reg_p2p.transformation)

    if is_visual:
        print("[INFO][heatmap::getDiffHeatMap]")
        print("\t start show icp result 1...")
        o3d.visualization.draw_geometries([partial_pointcloud_1, complete_mesh])
        print("[INFO][heatmap::getDiffHeatMap]")
        print("\t start show icp result 2...")
        o3d.visualization.draw_geometries([partial_pointcloud_2, complete_mesh])

    if partial_noise_sigma > 0:
        partial_points = np.array(partial_pointcloud_1.points)
        noise = getNoise(partial_points.shape[0], partial_noise_sigma, print_progress)
        partial_points += noise
        partial_pointcloud_1.points = o3d.utility.Vector3dVector(partial_points)

        partial_points = np.array(partial_pointcloud_2.points)
        noise = getNoise(partial_points.shape[0], partial_noise_sigma, print_progress)
        partial_points += noise
        partial_pointcloud_2.points = o3d.utility.Vector3dVector(partial_points)

    dist_to_partial_1 = np.array(complete_pointcloud.compute_point_cloud_distance(partial_pointcloud_1))
    dist_to_partial_2 = np.array(complete_pointcloud.compute_point_cloud_distance(partial_pointcloud_2))

    dist_list = []
    for_data = range(len(dist_to_partial_1))
    if print_progress:
        print("[INFO][heatmap::getDiffHeatMap]")
        print("\t start generate dist diff list...")
        for_data = tqdm(for_data)
    for i in for_data:
        current_dist_diff = max(0.0, dist_to_partial_2[i] - dist_to_partial_1[i])
        dist_list.append(current_dist_diff)

    colors = getDistColorMap(dist_list, color_map, error_max, print_progress)

    complete_pointcloud.colors = o3d.utility.Vector3dVector(colors)
    complete_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    complete_pointcloud.normals = o3d.utility.Vector3dVector()

    complete_mesh.compute_vertex_normals()

    #  sphere_complete_pointcloud = getSpherePointCloud(complete_pointcloud,
                                                     #  0.001,
                                                     #  20,
                                                     #  print_progress)

    if is_visual:
        print("[INFO][heatmap::getDiffHeatMap]")
        print("\t start show heatmap result 1...")
        o3d.visualization.draw_geometries([partial_pointcloud_1, complete_mesh])
        print("[INFO][heatmap::getDiffHeatMap]")
        print("\t start show heatmap result 2...")
        o3d.visualization.draw_geometries([partial_pointcloud_2, complete_mesh])

    if print_progress:
        print("[INFO][heatmap::getDiffHeatMap]")
        print("\t start save heatmap...")
    o3d.io.write_triangle_mesh(save_complete_mesh_file_path, complete_mesh,
                               write_ascii=True, print_progress=print_progress)
    return True

