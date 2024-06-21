#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import open3d as o3d

def transPointCloudFormat(source_pointcloud_file_path,
                          target_pointcloud_file_path,
                          print_progress=False,
                          need_estimate_normals=True,
                          estimate_normals_radius=0.05,
                          estimate_normals_max_nn=30):
    if not os.path.exists(source_pointcloud_file_path):
        print("[ERROR][trans::transPointCloudFormat]")
        print("source_pointcloud_file not exist!")
        return False

    pointcloud = o3d.io.read_point_cloud(source_pointcloud_file_path, print_progress=True)
    if need_estimate_normals:
        pointcloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=estimate_normals_radius,
                max_nn=estimate_normals_max_nn))

    o3d.io.write_point_cloud(
        target_pointcloud_file_path,
        pointcloud,
        write_ascii=True,
        print_progress=print_progress)
    return True

def transMeshFormat(source_mesh_file_path,
                    target_mesh_file_path,
                    print_progress=False,
                    need_estimate_normals=True):
    if not os.path.exists(source_mesh_file_path):
        print("[ERROR][trans::transMeshFormat]")
        print("source_mesh_file not exist!")
        return False

    mesh = o3d.io.read_triangle_mesh(source_mesh_file_path, print_progress=True)
    if need_estimate_normals:
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()

    o3d.io.write_triangle_mesh(
        target_mesh_file_path,
        mesh,
        write_ascii=True,
        print_progress=print_progress)
    return True

def transFormat(source_file_path,
                target_file_path,
                load_point_only=False,
                print_progress=False,
                need_estimate_normals=True,
                estimate_normals_radius=0.05,
                estimate_normals_max_nn=30):
    if load_point_only:
        return transPointCloudFormat(
            source_file_path, target_file_path, print_progress,
            need_estimate_normals, estimate_normals_radius, estimate_normals_max_nn)

    return transMeshFormat(
        source_file_path, target_file_path, print_progress, need_estimate_normals)

