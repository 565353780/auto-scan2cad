#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

from habitat_sim_manage.Data.rad import Rad

from habitat_sim_manage.Method.rotations import getDirectionFromRad

def getInversePose(pose, radius):
    direction = getDirectionFromRad(pose.rad)
    move_dist = direction.scale(radius)

    inverse_pose = deepcopy(pose)
    inverse_pose.position.add(move_dist)
    inverse_pose.rad = inverse_pose.rad.inverse()
    return inverse_pose

def getCenterRotatePose(pose,
                        radius,
                        up_rotate_angle,
                        right_rotate_angle,
                        front_rotate_angle):
    center_pose = getInversePose(pose, radius)

    rotate_rad = Rad(
        np.deg2rad(-up_rotate_angle),
        np.deg2rad(right_rotate_angle),
        np.deg2rad(-front_rotate_angle))
    center_pose.rad.add(rotate_rad)

    new_pose = getInversePose(center_pose, radius)
    return new_pose

def getCircleTurnLeftPose(pose, radius, rotate_angle):
    return getCenterRotatePose(pose, radius, rotate_angle, 0.0, 0.0)

def getCircleTurnRightPose(pose, radius, rotate_angle):
    return getCenterRotatePose(pose, radius, -rotate_angle, 0.0, 0.0)

def getCircleTurnUpPose(pose, radius, rotate_angle):
    return getCenterRotatePose(pose, radius, 0.0, rotate_angle, 0.0)

def getCircleTurnDownPose(pose, radius, rotate_angle):
    return getCenterRotatePose(pose, radius, 0.0, -rotate_angle, 0.0)

def getCircleHeadLeftPose(pose, radius, rotate_angle):
    return getCenterRotatePose(pose, radius, 0.0, 0.0, -rotate_angle)

def getCircleHeadRightPose(pose, radius, rotate_angle):
    return getCenterRotatePose(pose, radius, 0.0, 0.0, rotate_angle)

