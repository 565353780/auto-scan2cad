#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

from scan2cad_dataset_manage.Data.point import Point

inf = float("inf")


class BBox(object):

    def __init__(self,
                 min_point=Point(inf, inf, inf),
                 max_point=Point(-inf, -inf, -inf)):
        self.min_point = min_point
        self.max_point = max_point

        self.diff_point = Point(-inf, -inf, -inf)

        self.updateDiffPoint()
        return

    @classmethod
    def fromList(cls, bbox_list):
        bbox = cls(Point.fromList(bbox_list[0]), Point.fromList(bbox_list[1]))
        return bbox

    def toList(self):
        return [self.min_point.toList(), self.max_point.toList()]

    def isValid(self):
        return self.min_point.x != inf

    def updateDiffPoint(self):
        if not self.isValid():
            self.diff_point = Point(-inf, -inf, -inf)
            return True
        self.diff_point = Point(self.max_point.x - self.min_point.x,
                                self.max_point.y - self.min_point.y,
                                self.max_point.z - self.min_point.z)
        return True

    def addPoint(self, point):
        if not self.isValid():
            self.min_point = deepcopy(point)
            self.max_point = deepcopy(point)
            self.updateDiffPoint()
            return True

        self.min_point.x = min(self.min_point.x, point.x)
        self.min_point.y = min(self.min_point.y, point.y)
        self.min_point.z = min(self.min_point.z, point.z)
        self.max_point.x = max(self.max_point.x, point.x)
        self.max_point.y = max(self.max_point.y, point.y)
        self.max_point.z = max(self.max_point.z, point.z)
        self.updateDiffPoint()
        return True

    def addBBox(self, bbox):
        self.addPoint(bbox.min_point)
        self.addPoint(bbox.max_point)
        self.updateDiffPoint()
        return True

    def isInBBox(self, point):
        if point.x < self.min_point.x or point.x > self.max_point.x:
            return False
        if point.y < self.min_point.y or point.y > self.max_point.y:
            return False
        if point.z < self.min_point.z or point.z > self.max_point.z:
            return False
        return True

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[BBox]")
        print(line_start + "\t min_point =")
        self.min_point.outputInfo(info_level + 1)
        print(line_start + "\t max_point =")
        self.max_point.outputInfo(info_level + 1)
        print(line_start + "\t diff_point =")
        self.diff_point.outputInfo(info_level + 1)
        return True
