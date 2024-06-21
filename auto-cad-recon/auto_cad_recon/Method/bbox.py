#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def getMoveToNOCTrans(bbox, center):
    bbox_euler_angle = np.array([0.0, 0.0, 0.0])
    bbox = bbox.reshape(2, 3)
    bbox_center = np.mean(bbox, axis=0)
    bbox_translate = np.mean([bbox_center, center], axis=0) * -1.0
    bbox_diff = bbox[1] - bbox[0]
    bbox_scale = np.array([1.0 / np.max(bbox_diff) for _ in range(3)])
    return bbox_translate, bbox_euler_angle, bbox_scale


def getOBBFromABB(abb):
    x_min, y_min, z_min, x_max, y_max, z_max = abb
    obb = np.array([[x_min, y_min, z_min], [x_min, y_min, z_max],
                    [x_min, y_max, z_min], [x_min, y_max, z_max],
                    [x_max, y_min, z_min], [x_max, y_min, z_max],
                    [x_max, y_max, z_min], [x_max, y_max, z_max]])
    return obb
