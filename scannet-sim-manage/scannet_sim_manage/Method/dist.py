#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import sqrt
from copy import deepcopy


def getPointDist2(point_1, point_2):
    x_diff = point_1.x - point_2.x
    y_diff = point_1.y - point_2.y
    z_diff = point_1.z - point_2.z
    return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff


def getPointDist(point_1, point_2):
    dist2 = getPointDist2(point_1, point_2)
    return sqrt(dist2)


def getBBoxDist2(bbox_1, bbox_2):
    if not bbox_1.isValid() or not bbox_2.isValid():
        return float('inf')

    min_point_dist2 = getPointDist2(bbox_1.min_point, bbox_2.min_point)
    max_point_dist2 = getPointDist2(bbox_1.max_point, bbox_2.max_point)
    return min_point_dist2 + max_point_dist2


def getBBoxDist(bbox_1, bbox_2):
    bbox_dist2 = getBBoxDist2(bbox_1, bbox_2)
    return sqrt(bbox_dist2)


def getMatchListFromMatrix(matrix):
    inf = float("inf")

    matrix_copy = deepcopy(matrix)

    match_list = []
    match_value_list = []

    matrix_min_value = np.min(matrix_copy)
    while matrix_min_value != inf:
        if matrix_min_value > 1:
            break

        min_idx = np.where(matrix_copy == matrix_min_value)
        row = min_idx[0][0]
        col = min_idx[1][0]

        if len(match_value_list) > 0:
            match_value_mean = np.mean(match_value_list)

            new_match_value = matrix_copy[row][col]

            if new_match_value > match_value_mean * 10:
                break

        match_list.append([row, col])
        match_value_list.append(matrix_copy[row][col])

        matrix_copy[row, :] = inf
        matrix_copy[:, col] = inf

        matrix_min_value = np.min(matrix_copy)
    return match_list
