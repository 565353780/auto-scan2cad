#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scannet_sim_manage.Data.bbox import BBox
from scannet_sim_manage.Data.point import Point

from scannet_sim_manage.Method.dist import getBBoxDist2


class BBoxGroup(object):
    def __init__(self):
        self.bbox_list = []
        self.merge_bbox = BBox()
        self.mean_bbox = BBox()
        return

    def reset(self):
        self.bbox_list = []
        self.merge_bbox = BBox()
        self.mean_bbox = BBox()
        return True

    def updateMeanBBox(self):
        if len(self.bbox_list) == 0:
            self.mean_bbox = BBox()
            return True

        min_point = Point()
        max_point = Point()
        for bbox in self.bbox_list:
            min_point.add(bbox.min_point)
            max_point.add(bbox.max_point)

        min_point_mean = min_point.scale(len(self.bbox_list))
        max_point_mean = max_point.scale(len(self.bbox_list))

        self.mean_bbox = BBox(min_point_mean, max_point_mean)
        return True

    def addBBox(self, bbox):
        self.bbox_list.append(bbox)
        self.merge_bbox.addBBox(bbox)
        self.updateMeanBBox()
        return True

    def getBBoxDist(self, bbox):
        merge_bbox_dist = getBBoxDist2(self.merge_bbox, bbox)
        mean_bbox_dist = getBBoxDist2(self.mean_bbox, bbox)
        return min(merge_bbox_dist, mean_bbox_dist)
