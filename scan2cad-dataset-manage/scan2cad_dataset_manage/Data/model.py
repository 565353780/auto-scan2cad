#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scan2cad_dataset_manage.Data.trans import Trans
from scan2cad_dataset_manage.Data.point import Point
from scan2cad_dataset_manage.Data.bbox import BBox

from scan2cad_dataset_manage.Method.bbox import getTransBBoxArray


class Model(object):

    def __init__(self, model_dict=None, trans_world_to_scan_matrix=None):
        self.trans_model_to_world = None
        self.bbox = None
        self.center = None
        self.sym = None
        self.cad_id = None
        self.cad_cat_id = None
        self.cad_keypoint_list = None
        self.scan_keypoint_list = None

        self.trans_model_to_scan_matrix = None
        self.trans_bbox_array = None

        if model_dict is not None and trans_world_to_scan_matrix is not None:
            self.loadModelDict(model_dict, trans_world_to_scan_matrix)
        return

    def loadTrans(self, model_dict, trans_world_to_scan_matrix):
        self.trans_model_to_world = Trans(
            model_dict['trs']['translation'],
            model_dict['trs']['rotation'],
            model_dict['trs']['scale'],
        )

        trans_model_to_world_matrix = \
            self.trans_model_to_world.getTransMatrix()
        self.trans_model_to_scan_matrix = np.matmul(
            trans_world_to_scan_matrix, trans_model_to_world_matrix)
        return True

    def loadBBox(self, model_dict):
        self.center = Point.fromList(model_dict['center'])

        bbox_length = model_dict['bbox']
        self.bbox = BBox(
            Point(self.center.x - bbox_length[0],
                  self.center.y - bbox_length[1],
                  self.center.z - bbox_length[2]),
            Point(self.center.x + bbox_length[0],
                  self.center.y + bbox_length[1],
                  self.center.z + bbox_length[2]),
        )
        return True

    def loadSym(self, model_dict):
        self.sym = model_dict['sym'].split("__SYM_")[1]
        return True

    def loadCAD(self, model_dict):
        self.cad_id = model_dict['id_cad']
        self.cad_cat_id = model_dict['catid_cad']
        return True

    def loadKeyPoints(self, model_dict):
        self.cad_keypoint_list = model_dict['keypoints_cad']['position']
        self.scan_keypoint_list = model_dict['keypoints_scan']['position']
        return True

    def updateTransBBoxArray(self):
        self.trans_bbox_array = getTransBBoxArray(
            self.bbox, self.trans_model_to_scan_matrix)
        return True

    def loadModelDict(self, model_dict, trans_world_to_scan_matrix):
        assert self.loadTrans(model_dict, trans_world_to_scan_matrix)
        assert self.loadBBox(model_dict)
        assert self.loadSym(model_dict)
        assert self.loadCAD(model_dict)
        assert self.loadKeyPoints(model_dict)
        assert self.updateTransBBoxArray()
        return True

    def toDict(self):
        model_dict = {
            "bbox": self.bbox.toList(),
            "center": self.center.toList(),
            "sym": self.sym,
            "cad_id": self.cad_id,
            "cad_cat_id": self.cad_cat_id,
            "trans_matrix": self.trans_model_to_scan_matrix.tolist(),
            "trans_bbox_array": self.trans_bbox_array.tolist()
        }
        return model_dict
