#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

from scannet_sim_manage.Config.bbox import POINTS, LINES, COLORS

from scannet_sim_manage.Data.point import Point
from scannet_sim_manage.Data.bbox import BBox


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
