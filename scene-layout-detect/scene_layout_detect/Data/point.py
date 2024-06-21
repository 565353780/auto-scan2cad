#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Point(object):

    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
        return

    def isZero(self):
        if self.x == 0 and self.y == 0:
            return True
        return False

    def isFinite(self):
        inf_list = [float("inf"), -float("inf")]
        if self.x in inf_list or self.y in inf_list:
            return False
        return True

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[Point]")
        print(line_start + "\t [" + \
              str(self.x) + ", " + \
              str(self.y) + "]")
        return True
