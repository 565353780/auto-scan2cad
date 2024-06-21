#!/usr/bin/env python
# -*- coding: utf-8 -*-

from noc_transform.Data.obb import OBB
from noc_transform.Method.noc import getNOCOBB
from noc_transform.Method.transform import getNOCTransform


class TransformGenerator(object):

    def __init__(self):
        return

    def getOBB(self, obb_points):
        return OBB(obb_points)

    def getOBBFromABBPoints(self, min_point, max_point):
        return OBB.fromABBPoints(min_point, max_point)

    def getOBBFromABBList(self, abb_list):
        return OBB.fromABBList(abb_list)

    def getNOCOBB(self, obb):
        return getNOCOBB(obb)

    def getNOCTransform(self, obb):
        return getNOCTransform(obb)

    def getBoxTransform(self, obb):
        noc_obb = OBB.fromABBList([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5])
        return getNOCTransform(obb, noc_obb)
