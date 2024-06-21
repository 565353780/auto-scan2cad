#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

from scan2cad_dataset_manage.Config.bbox import POINTS, LINES, COLORS

from scan2cad_dataset_manage.Data.point import Point
from scan2cad_dataset_manage.Data.bbox import BBox


def getBBoxDist(bbox_1, bbox_2):
    min_x_diff = bbox_1.min_point.x - bbox_2.min_point.x
    min_y_diff = bbox_1.min_point.y - bbox_2.min_point.y
    min_z_diff = bbox_1.min_point.z - bbox_2.min_point.z
    max_x_diff = bbox_1.max_point.x - bbox_2.max_point.x
    max_y_diff = bbox_1.max_point.y - bbox_2.max_point.y
    max_z_diff = bbox_1.max_point.z - bbox_2.max_point.z

    bbox_dist = \
        min_x_diff * min_x_diff + \
        min_y_diff * min_y_diff + \
        min_z_diff * min_z_diff + \
        max_x_diff * max_x_diff + \
        max_y_diff * max_y_diff + \
        max_z_diff * max_z_diff
    return bbox_dist


def getNearestModelIdxByBBoxDist(bbox, scene):
    min_bbox_dist = float('inf')
    min_bbox_dist_model_idx = -1
    for i, model in enumerate(scene.model_list):
        current_bbox_dist = getBBoxDist(bbox, model.bbox)
        if current_bbox_dist < min_bbox_dist:
            min_bbox_dist = current_bbox_dist
            min_bbox_dist_model_idx = i
    return min_bbox_dist_model_idx


def getBBoxPointList(bbox):
    bbox_point_list = [
        [bbox.min_point.x, bbox.min_point.y, bbox.min_point.z, 1],
        [bbox.min_point.x, bbox.min_point.y, bbox.max_point.z, 1],
        [bbox.min_point.x, bbox.max_point.y, bbox.min_point.z, 1],
        [bbox.min_point.x, bbox.max_point.y, bbox.max_point.z, 1],
        [bbox.max_point.x, bbox.min_point.y, bbox.min_point.z, 1],
        [bbox.max_point.x, bbox.min_point.y, bbox.max_point.z, 1],
        [bbox.max_point.x, bbox.max_point.y, bbox.min_point.z, 1],
        [bbox.max_point.x, bbox.max_point.y, bbox.max_point.z, 1]
    ]
    return bbox_point_list


def getTransBBox(bbox, trans_matrix):
    bbox_point_list = getBBoxPointList(bbox)
    bbox_point_array = np.array(bbox_point_list).transpose(1, 0)
    trans_bbox_point_array = np.matmul(trans_matrix,
                                       bbox_point_array).transpose(1, 0)[:, :3]

    min_point_list = [np.min(trans_bbox_point_array[:, i]) for i in range(3)]
    max_point_list = [np.max(trans_bbox_point_array[:, i]) for i in range(3)]

    trans_bbox = BBox.fromList([min_point_list, max_point_list])
    return trans_bbox


def getTransBBoxArray(bbox, trans_matrix):
    bbox_point_list = getBBoxPointList(bbox)
    bbox_point_array = np.array(bbox_point_list).transpose(1, 0)
    trans_bbox_point_array = np.matmul(trans_matrix,
                                       bbox_point_array).transpose(1, 0)[:, :3]
    return trans_bbox_point_array


def getOpen3DBBox():
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(POINTS),
                                    lines=o3d.utility.Vector2iVector(LINES))
    line_set.colors = o3d.utility.Vector3dVector(COLORS)
    return line_set


def getBBoxFromOpen3DBBox(open3d_bbox):
    max_bound = open3d_bbox.get_max_bound()
    min_bound = open3d_bbox.get_min_bound()

    min_point = Point(min_bound[0], min_bound[1], min_bound[2])
    max_point = Point(max_bound[0], max_bound[1], max_bound[2])

    bbox = BBox()
    bbox.addPoint(min_point)
    bbox.addPoint(max_point)
    return bbox


def getOpen3DBBoxFromBBox(bbox, color=[255, 0, 0]):
    min_point_list = bbox.min_point.toList()
    max_point_list = bbox.max_point.toList()
    points = np.array([min_point_list, max_point_list])
    points = o3d.utility.Vector3dVector(points)

    open3d_bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
        points)
    open3d_bbox.color = np.array(color, dtype=np.float32) / 255.0
    return open3d_bbox


def getOpen3DBBoxFromBBoxArray(bbox_array, color=[255, 0, 0]):
    colors = np.array([color for _ in LINES], dtype=float) / 255.0
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(bbox_array),
        lines=o3d.utility.Vector2iVector(LINES))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set
