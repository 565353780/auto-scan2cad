#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mesh_manage.Data.point import Point

from mesh_manage.Method.dist import getPointDist


class BBox(object):

    def __init__(self, min_point=Point(), max_point=Point()):
        self.min_point = min_point
        self.max_point = max_point
        return

    @classmethod
    def fromList(cls, bbox_list):
        return cls(Point.fromList(bbox_list[0]), Point.fromList(bbox_list[1]))

    def toList(self):
        return [self.min_point.toList(), self.max_point.toList()]

    def isInBBox(self, point):
        if point.x < self.min_point.x or point.x > self.max_point.x:
            return False
        if point.y < self.min_point.y or point.y > self.max_point.y:
            return False
        if point.z < self.min_point.z or point.z > self.max_point.z:
            return False
        return True

    def getDistToCenter(self, point):
        center_point = Point((self.min_point.x + self.max_point.x) / 2.0,
                             (self.min_point.y + self.max_point.y) / 2.0,
                             (self.min_point.z + self.max_point.z) / 2.0)

        return getPointDist(point, center_point)

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[BBox]")
        print(line_start + "\t min_point:")
        self.min_point.outputInfo(info_level + 1)
        print(line_start + "\t max_point:")
        self.max_point.outputInfo(info_level + 1)
        return True
