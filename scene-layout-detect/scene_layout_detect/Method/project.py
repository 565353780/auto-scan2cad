#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy


def getProjectPoints(point_array):
    points = point_array[np.where(point_array[:, 0] != float("inf"))[0]]
    project_points = deepcopy(points)

    project_points[:, 2] = 0
    return project_points
