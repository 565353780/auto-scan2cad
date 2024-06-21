#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from habitat_sim_manage.Config.angle import PI, TWO_PI, HALF_PI


class Rad(object):

    def __init__(self,
                 up_rotate_rad=0.0,
                 right_rotate_rad=0.0,
                 front_rotate_rad=0.0):
        self.up_rotate_rad = up_rotate_rad
        self.right_rotate_rad = right_rotate_rad
        self.front_rotate_rad = front_rotate_rad

        self.update()
        return

    @classmethod
    def fromList(cls, urf_list):
        return cls(urf_list[0], urf_list[1], urf_list[2])

    def update(self):
        while self.up_rotate_rad < -PI:
            self.up_rotate_rad += TWO_PI
        while self.up_rotate_rad >= PI:
            self.up_rotate_rad -= TWO_PI

        if self.right_rotate_rad < -HALF_PI:
            self.right_rotate_rad = -HALF_PI
        if self.right_rotate_rad > HALF_PI:
            self.right_rotate_rad = HALF_PI

        while self.front_rotate_rad < 0:
            self.front_rotate_rad += TWO_PI
        while self.front_rotate_rad >= TWO_PI:
            self.front_rotate_rad -= TWO_PI
        return True

    def inverse(self):
        return Rad(self.up_rotate_rad + PI, -self.right_rotate_rad,
                   -self.front_rotate_rad)

    def add(self, rad):
        self.up_rotate_rad += rad.up_rotate_rad
        self.right_rotate_rad += rad.right_rotate_rad
        self.front_rotate_rad += rad.front_rotate_rad

        self.update()
        return True

    def toList(self):
        return [
            self.up_rotate_rad, self.right_rotate_rad, self.front_rotate_rad
        ]

    def toArray(self):
        point_list = self.toList()
        return np.array(point_list, dtype=np.float32)

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level

        print(line_start + "[Rad]")
        print(line_start + "\t up_rotate_angle = " + \
              str(np.rad2deg(self.up_rotate_rad)))
        print(line_start + "\t right_rotate_angle = " + \
              str(np.rad2deg(self.right_rotate_rad)))
        print(line_start + "\t front_rotate_angle = " + \
              str(np.rad2deg(self.front_rotate_rad)))
        return True
