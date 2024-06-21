#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scan2cad_dataset_manage.Data.scene import Scene


class Dataset(object):

    def __init__(self, scene_dict_list=None):
        self.scene_dict = {}

        if scene_dict_list is not None:
            self.loadSceneDictList(scene_dict_list)
        return

    def loadSceneDictList(self, scene_dict_list):
        for scene_dict in scene_dict_list:
            scene_name = scene_dict['id_scan']
            self.scene_dict[scene_name] = Scene(scene_dict)
        return True
