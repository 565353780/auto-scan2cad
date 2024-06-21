#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import sqrt, atan, pi

from scene_layout_detect.Data.point import Point


class Line(object):

    def __init__(self, start=Point(), end=Point()):
        self.start = start
        self.end = end

        self.diff_point = None
        self.k = None
        self.update()
        return

    def updateDiffPoint(self):
        x_diff = self.end.x - self.start.x
        y_diff = self.end.y - self.start.y
        self.diff_point = Point(x_diff, y_diff)
        return True

    def updateK(self):
        if self.diff_point.x == 0:
            if self.diff_point.y == 0:
                self.k = None
                return True

            self.k = float("inf")
            return True

        self.k = self.diff_point.y / self.diff_point.x
        return True

    def update(self):
        if not self.updateDiffPoint():
            print("[ERROR][Line::update]")
            print("\t updateDiffPoint failed!")
            return False
        if not self.updateK():
            print("[ERROR][Line::update]")
            print("\t updateK failed!")
            return False
        return True

    def getLength(self):
        x_diff = self.end.x - self.start.x
        y_diff = self.end.y - self.start.y
        length2 = x_diff * x_diff + y_diff * y_diff
        return sqrt(length2)

    def getRad(self):
        if self.k is None:
            return None

        rad = atan(self.k)
        return rad

    def getAngle(self):
        if self.k is None:
            return None

        rad = atan(self.k)
        angle = rad * 180.0 / pi
        return angle

    def isPoint(self):
        if self.diff_point.x == 0 and self.diff_point.y == 0:
            return True
        return False

    def getMiddlePoint(self):
        x_mean = (self.start.x + self.end.x) / 2.0
        y_mean = (self.start.y + self.end.y) / 2.0
        middle_point = Point(x_mean, y_mean)
        return middle_point

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[Line]")
        print(line_start + "\t start:")
        self.start.outputInfo(info_level + 1)
        print(line_start + "\t end:")
        self.end.outputInfo(info_level + 1)
        print(line_start + "\t diff_point :")
        self.diff_point.outputInfo(info_level + 1)
        print(line_start + "\t k =", self.k)
        return True
