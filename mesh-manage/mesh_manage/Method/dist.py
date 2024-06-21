#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt

def getPointDist2(point_1, point_2):
    x_diff = point_2.x - point_1.x
    y_diff = point_2.y - point_1.y
    z_diff = point_2.z - point_1.z
    dist2 = x_diff * x_diff + y_diff * y_diff + z_diff * z_diff
    return dist2

def getPointDist(point_1, point_2):
    dist2 = getPointDist2(point_1, point_2)
    dist = sqrt(dist2)
    return dist
