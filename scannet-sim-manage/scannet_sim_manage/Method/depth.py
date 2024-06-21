#!/usr/bin/env python
# -*- coding: utf-8 -*-

import quaternion
import numpy as np
from copy import deepcopy

from scannet_sim_manage.Config.depth import K_INV, XS, YS


def getGaussNoise(depth_image, mean=0, var=0.0005):
    noise = np.random.normal(mean, var**0.5, depth_image.shape)
    zero_pixel_idx = np.where(depth_image == 0)
    noise[zero_pixel_idx] = 0

    new_depth_image = depth_image + noise
    new_depth_image = np.clip(new_depth_image, 0, float('inf'))
    return new_depth_image


def getDepthPoint(depth_obs):
    depth = depth_obs.reshape(1, depth_obs.shape[1], depth_obs.shape[0])
    xys = np.vstack((XS * depth, YS * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)

    xy_c0 = np.matmul(K_INV, xys)
    return xy_c0


def getCameraPoint(observations):
    depth_obs = observations["depth_sensor"]
    #  depth_obs = deepcopy(observations["depth_sensor"])

    depth_obs[np.where(depth_obs > 2)] = 0
    depth_obs = getGaussNoise(depth_obs)
    return getDepthPoint(depth_obs)


def getCameraBoundary(observations, boundary_length=0.5):
    depth_obs = np.zeros_like(observations["depth_sensor"])
    width, height = depth_obs.shape
    depth_obs[0][0] = boundary_length
    depth_obs[0][height - 1] = boundary_length
    depth_obs[width - 1][0] = boundary_length
    depth_obs[width - 1][height - 1] = boundary_length
    return getDepthPoint(depth_obs)


def getCameraToWorldMatrix(agent_state):
    rotation = agent_state.sensor_states['depth_sensor'].rotation
    translation = agent_state.sensor_states['depth_sensor'].position
    rotation = quaternion.as_rotation_matrix(rotation)
    T_camera_world = np.eye(4)
    T_camera_world[0:3, 0:3] = rotation
    T_camera_world[0:3, 3] = translation
    return T_camera_world


def getPointArray(observations, agent_state, boundary_length=0.5):
    T_camera_world = getCameraToWorldMatrix(agent_state)

    xy_c0 = getCameraPoint(observations)

    point_array = np.matmul(T_camera_world,
                            xy_c0)[:3, :].transpose(1, 0)[..., [0, 2, 1]]
    point_array[:, 1] *= -1

    xy_boundary = getCameraBoundary(observations, boundary_length)

    boundary_point_array = np.matmul(
        T_camera_world, xy_boundary)[:3, :].transpose(1, 0)[..., [0, 2, 1]]
    boundary_point_array[:, 1] *= -1

    camera_point = np.matmul(T_camera_world,
                             [[0], [0], [0], [1]])[:3][[0, 2, 1]]
    camera_point[1] *= -1

    camera_face_to_point = np.matmul(T_camera_world,
                                     [[0], [0], [-1], [1]])[:3][[0, 2, 1]]
    camera_face_to_point[1] *= -1

    match_camera_idx = (boundary_point_array[:, 0] == camera_point[0]) & (
        boundary_point_array[:, 1]
        == camera_point[1]) & (boundary_point_array[:, 2] == camera_point[2])
    valid_idx = np.where(match_camera_idx == False)[0]
    valid_boundary_point_array = boundary_point_array[valid_idx]

    return point_array, camera_point, valid_boundary_point_array, camera_face_to_point
