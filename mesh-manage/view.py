#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mesh_manage.Method.render import render

# Param
pointcloud_file_path = "/home/chli/.ros/RUN_LOG/PointCloud2ToObjectVecConverterServer/2022_9_2_12-53-42/scene_10.pcd"
estimate_normals_radius = 0.05
estimate_normals_max_nn = 30

# Process
render(pointcloud_file_path, estimate_normals_radius, estimate_normals_max_nn)

