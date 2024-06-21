#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from tqdm import tqdm

source_ply_file_path = "/home/chli/chLi/NeRF/ustc_niu/dense/dense/0/fused.ply"

target_ply_file_path = "/home/chli/chLi/NeRF/ustc_niu/dense/dense/0/target.ply"

pcd = o3d.io.read_point_cloud(source_ply_file_path)

points = np.array(pcd.points)
colors = np.array(pcd.colors)

white_color_num = 0
for color in tqdm(colors):
    if (color == [1.0, 1.0, 1.0]).all():
        white_color_num += 1
print(white_color_num)

white_thresh = 0.7
x_valid_idx = np.where(colors[:, 0] < white_thresh)[0]
valid_x_colors = colors[x_valid_idx]
xy_valid_idx = x_valid_idx[np.where(valid_x_colors[:, 1] < white_thresh)[0]]
valid_xy_colors = colors[xy_valid_idx]
xyz_valid_idx = xy_valid_idx[np.where(valid_xy_colors[:, 2] < white_thresh)[0]]

new_points = points[xyz_valid_idx]
new_colors = colors[xyz_valid_idx]

print(new_points.shape)
print(points.shape[0] - new_points.shape[0])

pcd.points = o3d.utility.Vector3dVector(new_points)
pcd.colors = o3d.utility.Vector3dVector(new_colors)

#  o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud(target_ply_file_path, pcd, write_ascii=True)

