#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d

folder_path = "/home/chli/chLi/PoinTr/ShapeNet55/shapenet_pc/"
save_folder_path = "/home/chli/chLi/PoinTr/ShapeNet55/pcd/"
os.makedirs(save_folder_path, exist_ok=True)

file_list = os.listdir(folder_path)

for i, file in enumerate(file_list):
    file_path = folder_path + file
    points = np.load(file_path)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    print(file_path)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(save_folder_path + str(i) + ".ply",
                             pcd,
                             write_ascii=True)
