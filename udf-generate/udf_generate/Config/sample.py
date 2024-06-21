#!/usr/bin/env python
# -*- coding: utf-8 -*-

import open3d as o3d

from udf_generate.Method.samples import getSamplePointMatrix, getArrayFromMatrix

SAMPLE_NUM = 32

SAMPLE_Z_ANGLE_LIST = [-120, 0, 120]
SAMPLE_X_ANGLE_LIST = [-120, 0, 120]
SAMPLE_Y_ANGLE_LIST = [-120, 0, 120]

SAMPLE_POINT_MATRIX = getSamplePointMatrix(SAMPLE_NUM)

SAMPLE_POINT_CLOUD = o3d.geometry.PointCloud()
SAMPLE_POINT_CLOUD.points = o3d.utility.Vector3dVector(
    getArrayFromMatrix(SAMPLE_POINT_MATRIX, 3))
