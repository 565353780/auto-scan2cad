#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scannet_dataset_manage.Data.scene import Scene


class SceneLoader(object):

    def __init__(self, scene_folder_path):
        self.scene = Scene(scene_folder_path)
        return

    def getLabeledObjectNum(self):
        return self.scene.getLabeledObjectNum()

    def getPointIdxListByLabeledObjectId(self, labeled_object_id):
        labeled_object = self.scene.getLabeledObjectById(labeled_object_id)
        assert labeled_object is not None
        return self.scene.getPointIdxListByLabeledObject(labeled_object)
