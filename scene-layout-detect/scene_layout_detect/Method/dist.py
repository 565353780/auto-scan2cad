#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
from scipy import optimize


def fLine(xy, A, B, C):
    x = xy[:, 0]
    y = xy[:, 1]
    return A * x + B * y + C


def f_line(x, A, B):
    return A * x + B


def f_line_inv(y, A_inv, B_inv):
    return A_inv * y + B_inv


def getPointDistToPoint(point_1, point_2):
    point_1_array = np.array(point_1, dtype=float)
    point_2_array = np.array(point_2, dtype=float)
    dist = np.linalg.norm(point_1_array - point_2_array)
    return dist


def isSamePoint(point_list):
    point_num = len(point_list)

    if point_num < 2:
        return True

    first_point = point_list[0]

    for i in range(1, point_num):
        point = point_list[i]
        if point[0] != first_point[0] or point[1] != first_point[1]:
            return False
    return True


def fitLine(point_list):
    assert len(point_list) > 0

    if isSamePoint(point_list):
        return None, point_list[0][0], point_list[0][1]

    point_array = np.array(point_list, dtype=float)

    A_inv, B_inv = optimize.curve_fit(f_line, point_array[:, 1],
                                      point_array[:, 0])[0]

    if A_inv == 0:
        return 1.0, 0.0, -B_inv

    A, B = optimize.curve_fit(f_line, point_array[:, 0], point_array[:, 1])[0]

    if A > 0:
        return A, -1.0, B

    return -A, 1.0, -B


def getPointDistToLine(point, line_param):
    A, B, C = line_param

    if A is None:
        line_point = [B, C]

        return getPointDistToPoint(point, line_point)

    line_weight = A * A + B * B

    assert line_weight > 0

    distance = np.abs(A * point[0] + B * point[1] + C) / np.sqrt(line_weight)
    return distance


def getProjectPoint(point, line_param):
    A, B, C = line_param

    if A is None:
        return np.array([B, C], dtype=float)

    project_point = deepcopy(np.array(point, dtype=float))

    point_value = A * point[0] + B * point[1] + C

    if point_value == 0:
        return project_point

    line_weight = np.sqrt(A * A + B * B)

    move_dist = getPointDistToLine(point, line_param)

    if point_value > 0:
        move_direction = [-A / line_weight, -B / line_weight]
    else:
        move_direction = [A / line_weight, B / line_weight]

    project_point[0] = project_point[0] + move_direction[0] * move_dist
    project_point[1] = project_point[1] + move_direction[1] * move_dist
    return project_point
