#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../points-shape-detect")

import numpy as np
import open3d as o3d

from points_shape_detect.Method.trans import transPointArray

from noc_transform.Data.obb import OBB
from noc_transform.Module.transform_generator import TransformGenerator


def demo():
    transform_generator = TransformGenerator()

    obb = OBB.fromABBList([-0.4, -0.3, -0.5, 0.4, 0.3, 0.5])
    points = obb.points
    translate = np.array([3, 4, 3])
    euler_angle = np.array([0, 45, 0])
    scale = np.array([2, 2, 2])

    points = transPointArray(points, translate, euler_angle, scale)
    obb.points = points

    trans_matrix = transform_generator.getNOCTransform(obb)

    trans_obb = obb.clone()
    trans_obb.transform(trans_matrix)
    print(trans_obb.points)
    return True
