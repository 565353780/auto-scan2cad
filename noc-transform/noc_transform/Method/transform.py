#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

from noc_transform.Method.noc import getNOCOBB


def getNOCTransform(obb, noc_obb=None):
    if noc_obb is None:
        noc_obb = getNOCOBB(obb)

    source_points = np.ones((8, 4), dtype=float)
    source_points[:, :3] = deepcopy(obb.points)

    target_points = np.ones((8, 4), dtype=float)
    target_points[:, :3] = deepcopy(noc_obb.points)

    A = source_points.T.dot(source_points)
    b = source_points.T.dot(target_points)

    transform = np.linalg.inv(A).dot(b)
    return transform


def transPoints(points, transform):
    copy_points = np.ones((points.shape[0], 4))
    copy_points[:, :3] = deepcopy(points)
    copy_points = copy_points @ transform
    return copy_points[:, :3]
