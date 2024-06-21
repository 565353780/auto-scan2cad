#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


def fps(point_array, sample_num):
    if point_array.shape[0] <= sample_num:
        return point_array

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array)
    sample_pcd = pcd.farthest_point_down_sample(sample_num)
    points = np.array(sample_pcd.points)
    return points
