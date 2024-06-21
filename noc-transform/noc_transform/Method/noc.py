#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

from noc_transform.Config.lines import X_AXIS_LINES, Y_AXIS_LINES, Z_AXIS_LINES
from noc_transform.Data.obb import OBB


def getNOCLength(obb, method="mean"):
    assert method in ["mean", "max", "min"]

    if method == "mean":
        op = np.mean
    elif method == "max":
        op = np.max
    elif method == "min":
        op = np.min

    points = deepcopy(obb.points)

    x_axis_length_list = []
    for line_idx in X_AXIS_LINES:
        start_point = points[line_idx[0]]
        end_point = points[line_idx[1]]
        diff_point = end_point - start_point
        length = np.linalg.norm(diff_point)
        x_axis_length_list.append(length)

    y_axis_length_list = []
    for line_idx in Y_AXIS_LINES:
        start_point = points[line_idx[0]]
        end_point = points[line_idx[1]]
        diff_point = end_point - start_point
        length = np.linalg.norm(diff_point)
        y_axis_length_list.append(length)

    z_axis_length_list = []
    for line_idx in Z_AXIS_LINES:
        start_point = points[line_idx[0]]
        end_point = points[line_idx[1]]
        diff_point = end_point - start_point
        length = np.linalg.norm(diff_point)
        z_axis_length_list.append(length)

    x_axis_length = op(x_axis_length_list)
    y_axis_length = op(y_axis_length_list)
    z_axis_length = op(z_axis_length_list)

    obb_length = np.array([x_axis_length, y_axis_length, z_axis_length])

    max_axis_length = np.max(obb_length)

    noc_length = obb_length / max_axis_length
    return noc_length


def getNOCOBB(obb, method="mean"):
    noc_length = getNOCLength(obb, method)
    noc_half_length = noc_length / 2.0

    min_point = -noc_half_length
    max_point = noc_half_length
    return OBB.fromABBPoints(min_point, max_point)
