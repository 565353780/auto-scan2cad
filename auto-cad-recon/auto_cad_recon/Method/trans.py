#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy


def transPoints(points, transform):
    copy_points = np.ones((points.shape[0], 4))
    copy_points[:, :3] = deepcopy(points)
    copy_points = copy_points @ transform
    return copy_points[:, :3]
