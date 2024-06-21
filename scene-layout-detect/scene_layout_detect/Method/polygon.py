#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np

from scene_layout_detect.Method.angle import getInRangeAngleIdx, getPointsAngle
from scene_layout_detect.Method.project import getProjectPoints


def getPolygon(camera_point, point_array, delta_angle):
    point_list = []

    points = getProjectPoints(point_array)

    copy_camera_point = deepcopy(camera_point).reshape(3)
    copy_camera_point[2] = 0.0
    angle_array = getPointsAngle(copy_camera_point, points)

    point_list.append(copy_camera_point)

    mean_point = np.mean(points, axis=0)
    start_angle_vector = copy_camera_point - mean_point
    start_angle = np.arctan2(start_angle_vector[1],
                             start_angle_vector[0]) * 180.0 / np.pi

    angle_num = int(360.0 / delta_angle)
    delta_angle = 360.0 / angle_num
    for i in range(angle_num):
        angle_min = i * delta_angle + start_angle
        angle_max = (i + 1) * delta_angle + start_angle
        in_range_point_idx = getInRangeAngleIdx(angle_array, angle_min,
                                                angle_max)
        if in_range_point_idx.shape[0] == 0:
            continue

        angle_mean = (angle_min + angle_max) / 2.0
        rad_mean = angle_mean * np.pi / 180.0

        diff_array = points[in_range_point_idx] - copy_camera_point
        dist_array = np.linalg.norm(diff_array, ord=2, axis=1)
        max_dist = np.max(dist_array)

        max_dist_point = np.array([
            copy_camera_point[0] + np.cos(rad_mean) * max_dist,
            copy_camera_point[1] + np.sin(rad_mean) * max_dist, 0.0
        ])

        point_list.append(max_dist_point)

    polygon = np.array(point_list)
    return polygon
