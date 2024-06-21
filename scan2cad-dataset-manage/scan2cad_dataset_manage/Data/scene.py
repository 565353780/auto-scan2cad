#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scan2cad_dataset_manage.Data.trans import Trans
from scan2cad_dataset_manage.Data.model import Model


class Scene(object):

    def __init__(self, scene_dict=None):
        self.scene_name = None
        self.trans_scan_to_world = None
        self.model_list = []

        self.trans_world_to_scan_matrix = None

        if scene_dict is not None:
            self.loadSceneDict(scene_dict)
        return

    def loadTrans(self, scene_dict):
        self.trans_scan_to_world = Trans(scene_dict['trs']['translation'],
                                         scene_dict['trs']['rotation'],
                                         scene_dict['trs']['scale'])

        self.trans_world_to_scan_matrix = np.linalg.inv(
            self.trans_scan_to_world.getTransMatrix())
        return True

    def loadSceneDict(self, scene_dict):
        self.scene_name = scene_dict['id_scan']

        assert self.loadTrans(scene_dict)

        for model_dict in scene_dict['aligned_models']:
            self.model_list.append(
                Model(model_dict, self.trans_world_to_scan_matrix))
        return True
