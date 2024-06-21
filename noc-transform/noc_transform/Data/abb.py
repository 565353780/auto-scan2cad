#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class ABB(object):

    def __init__(self, min_point=[0.0, 0.0, 0.0], max_point=[0.0, 0.0, 0.0]):
        self.min_point = np.array(min_point, dtype=float)
        self.max_point = np.array(max_point, dtype=float)
        return

    @classmethod
    def fromList(cls, point_list):
        min_point = point_list[:3]
        max_point = point_list[3:]
        return cls(min_point, max_point)

    def toArray(self):
        return np.vstack((self.min_point, self.max_point))

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[ABB]")
        print(line_start + "\t min_point:")
        print(line_start + "\t\t", self.min_point)
        print(line_start + "\t max_point:")
        print(line_start + "\t\t", self.max_point)
        return True
