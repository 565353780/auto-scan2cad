#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy

from noc_transform.Data.abb import ABB


class OBB(object):

    def __init__(self, points):
        self.points = np.array(points)
        return

    @classmethod
    def fromABB(cls, abb):
        min_point, max_point = abb.toArray()
        points = [[min_point[0], min_point[1], min_point[2]],
                  [min_point[0], min_point[1], max_point[2]],
                  [min_point[0], max_point[1], min_point[2]],
                  [min_point[0], max_point[1], max_point[2]],
                  [max_point[0], min_point[1], min_point[2]],
                  [max_point[0], min_point[1], max_point[2]],
                  [max_point[0], max_point[1], min_point[2]],
                  [max_point[0], max_point[1], max_point[2]]]
        return cls(points)

    @classmethod
    def fromABBPoints(cls, min_point, max_point):
        abb_min_point = deepcopy(min_point)
        abb_max_point = deepcopy(max_point)
        abb = ABB(abb_min_point, abb_max_point)
        return cls.fromABB(abb)

    @classmethod
    def fromABBList(cls, abb_list):
        abb = ABB.fromList(abb_list)
        return cls.fromABB(abb)

    def clone(self):
        points = deepcopy(self.points)
        return OBB(points)

    def toArray(self):
        return self.points

    def toABB(self):
        min_point = deepcopy(self.points[0])
        max_point = deepcopy(self.points[7])
        return ABB(min_point, max_point)

    def toABBArray(self):
        return self.toABB().toArray()

    def transform(self, trans_matrix):
        points = np.ones((8, 4), dtype=float)
        points[:, :3] = self.points

        points = points @ trans_matrix
        self.points = points[:, :3]
        return True

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[OBB]")
        print(line_start + "\t min_point:")
        print(line_start + "\t\t", self.points[0])
        print(line_start + "\t max_point:")
        print(line_start + "\t\t", self.points[7])
        return True
