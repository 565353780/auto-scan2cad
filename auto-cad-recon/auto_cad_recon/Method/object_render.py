#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


def getObjectPCDList(data_list,
                     color_map=None,
                     translate=[0, 0, 0],
                     estimate_normals=False):
    translate = np.array(translate, dtype=float)
    if color_map is not None:
        color_map = np.array(color_map, dtype=float)

    pcd_list = []

    for i, data in enumerate(data_list):
        points = data['predictions']['merged_point_array']

        if color_map is None:
            colors = data['predictions']['merged_color_array'] / 255.0
        else:
            colors = np.zeros_like(points)
            colors[:] = color_map[i % color_map.shape[0]] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.translate(translate)

        if estimate_normals:
            pcd.estimate_normals()

        pcd_list.append(pcd)
    return pcd_list
