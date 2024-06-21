#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from quaternion import quaternion

from scan2cad_dataset_manage.Method.matrix import make_M_from_tqs


class Trans(object):

    def __init__(self,
                 translation=[0.0, 0.0, 0.0],
                 rotation=[0.0, 0.0, 0.0, 1.0],
                 scale=[1.0, 1.0, 1.0]):
        self.translation = np.array(translation)
        self.rotation = np.array(rotation)
        self.scale = np.array(scale)
        return

    def getTransMatrix(self):
        trans_matrix = make_M_from_tqs(self.translation.tolist(),
                                       self.rotation.tolist(),
                                       self.scale.tolist())
        return trans_matrix

    def getQuaternion(self):
        quat = quaternion(self.rotation[0], self.rotation[1], self.rotation[2],
                          self.rotation[3])
        return quat

    def toDict(self):
        trans_dict = {
            "translation": self.translation,
            "rotation": self.rotation,
            "scale": self.scale
        }
        return trans_dict

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level

        print(line_start + "[Trans]")
        print(line_start + "\t translation =", self.translation)
        print(line_start + "\t rotation =", self.rotation)
        print(line_start + "\t scale =", self.scale)
        return True
