#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scannet_sim_manage.Data.scene_object import SceneObject


class SceneObjectManager(object):

    def __init__(self):
        self.scene_object_dict = {}
        return

    def reset(self):
        del self.scene_object_dict
        self.scene_object_dict = {}
        return True

    def addFrameObject(self,
                       object_label,
                       frame_idx,
                       point_image,
                       label,
                       value=True):
        if object_label not in self.scene_object_dict.keys():
            self.scene_object_dict[object_label] = SceneObject(object_label)

        return self.scene_object_dict[object_label].addFrameObject(
            frame_idx, point_image, label, value)

    def getSceneObjectLabelList(self):
        return list(self.scene_object_dict.keys())

    def getFrameObject(self, object_label, frame_idx):
        assert object_label in self.scene_object_dict
        return self.scene_object_dict[object_label].getFrameObject(frame_idx)

    def getFrameObjectDict(self, object_label):
        assert object_label in self.scene_object_dict
        return self.scene_object_dict[object_label].frame_object_dict

    def extractObjectsFromPointImage(self, point_image, frame_idx):
        label_list = point_image.getValidLabelValueList()

        for label, value in label_list:
            self.addFrameObject(label + "==" + str(value), frame_idx,
                                point_image, label, value)
        return True
