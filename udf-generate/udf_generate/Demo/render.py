#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from udf_generate.Method.udfs import normalizeMesh, translateMesh, loadUDF, getVisualUDF, getPointUDF


def demo():
    mesh_file_path = \
        "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/03001627/bdc892547cceb2ef34dedfee80b7006/models/model_normalized.obj"
    pcd_file_path = \
        "/home/chli/github/auto-cad-recon/output/scene_objects/scene0474_02/1_chair.ply==object/merged_point_array.ply"
    udf_file_path = \
        "/home/chli/chLi/ShapeNet/udfs/02691156/222c0d99446148babe4274edc10c1c8e/models/model_normalized/udf.npy"

    mesh = o3d.io.read_triangle_mesh(mesh_file_path)
    normalizeMesh(mesh)
    translateMesh(mesh, 1.0, 0, 0)

    pcd = o3d.io.read_point_cloud(pcd_file_path)
    points = np.array(pcd.points)
    point_udf = getPointUDF(points)
    point_udf_pcd = getVisualUDF(point_udf)

    cad_udf = loadUDF(udf_file_path)

    cad_udf_pcd = getVisualUDF(cad_udf)
    o3d.visualization.draw_geometries([mesh, point_udf_pcd])
    return True
