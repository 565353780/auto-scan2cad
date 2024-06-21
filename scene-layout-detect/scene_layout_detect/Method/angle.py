#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def getValidAngle(angle):
    while angle < 0.0:
        angle += 360.0
    while angle >= 360.0:
        angle -= 360.0
    return angle


def getPointsAngle(start_point, point_array, return_angle=True):
    diff_point = point_array - start_point
    rad_array = np.arctan2(diff_point[:, 1], diff_point[:, 0])

    if not return_angle:
        return rad_array

    angle_array = rad_array * 180.0 / np.pi
    return angle_array


def getInRangeAngleIdx(angle_array, angle_min, angle_max):
    assert -180.0 < angle_min < angle_max < 360.0 + 180.0

    higher_angle_mask_1 = angle_array >= angle_min
    lower_angle_mask_1 = angle_array <= angle_max
    in_range_angle_mask_1 = higher_angle_mask_1 & lower_angle_mask_1

    higher_angle_mask_2 = angle_array >= angle_min - 360.0
    lower_angle_mask_2 = angle_array <= angle_max - 360.0
    in_range_angle_mask_2 = higher_angle_mask_2 & lower_angle_mask_2

    in_range_angle_mask = in_range_angle_mask_1 | in_range_angle_mask_2

    in_range_angle_idx = np.where(in_range_angle_mask == True)[0]
    return in_range_angle_idx
