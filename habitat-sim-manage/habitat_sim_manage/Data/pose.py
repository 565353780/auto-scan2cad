#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from habitat_sim_manage.Data.point import Point
from habitat_sim_manage.Data.rad import Rad


class Pose(object):

    def __init__(self, position=Point(), rad=Rad(), scale=[1.0, 1.0, 1.0]):
        self.position = position
        self.rad = rad
        self.scale = np.array(scale)
        return

    @classmethod
    def fromList(cls, xyz_list, urf_list, scale=[1.0, 1.0, 1.0]):
        return cls(Point.fromList(xyz_list), Rad.fromList(urf_list), scale)

    def toList(self):
        return [self.position.toList(), self.rad.toList(), self.scale.tolist()]

    def setPosition(self, position):
        self.position = position
        return True

    def setRad(self, rad):
        self.rad = rad
        return True

    def setScale(self, scale):
        self.scale = np.array(scale)
        return True

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level

        print(line_start + "[Pose]")
        self.position.outputInfo(info_level + 1)
        self.rad.outputInfo(info_level + 1)
        print(line_start + "\t scale =", self.scale)
        return True
