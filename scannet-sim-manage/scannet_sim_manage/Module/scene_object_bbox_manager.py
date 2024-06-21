#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scannet_sim_manage.Data.bbox_group import BBoxGroup

class SceneObjectBBoxManager(object):
    def __init__(self):
        self.bbox_group_dict = {}
        return

    def reset(self):
        self.bbox_group_dict = {}
        return True

    def addBBox(self, bbox):
        #TODO: finish add bbox and auto merge here
        return True
