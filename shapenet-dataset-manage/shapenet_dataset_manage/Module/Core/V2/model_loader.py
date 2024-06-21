#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shapenet_dataset_manage.Data.model import Model


class ModelLoader(object):

    def __init__(self):
        self.model = Model()
        return

    def reset(self):
        self.model.reset()
        return True

    def loadModel(self, model_root_path):
        assert self.model.loadRootPath(model_root_path)
        return True

    def outputInfo(self, info_level=0):
        self.model.outputInfo(info_level)
        return True
