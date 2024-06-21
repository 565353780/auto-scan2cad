#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

from scan2cad_dataset_manage.Method.matrix import make_M_from_tqs, decompose_mat4
from scan2cad_dataset_manage.Method.render import renderScan2CADObjectModelMap


class ObjectModelMapManager(object):

    def __init__(self,
                 scannet_object_dataset_folder_path=None,
                 shapenet_dataset_folder_path=None,
                 object_model_map_dataset_folder_path=None):

        self.scannet_object_dataset_folder_path = None
        self.shapenet_dataset_folder_path = None
        self.object_model_map_dataset_folder_path = None

        self.scene_name = None
        self.scene_name_list = None
        self.scene_object_model_map_dict = None

        if None not in [
                scannet_object_dataset_folder_path,
                shapenet_dataset_folder_path,
                object_model_map_dataset_folder_path
        ]:
            self.loadDataset(scannet_object_dataset_folder_path,
                             shapenet_dataset_folder_path,
                             object_model_map_dataset_folder_path)
        return

    def reset(self):
        self.scannet_object_dataset_folder_path = None
        self.shapenet_dataset_folder_path = None
        self.object_model_map_dataset_folder_path = None

        self.scene_name_list = None
        self.scene_object_model_map_dict = None
        return True

    def loadDataset(self, scannet_object_dataset_folder_path,
                    shapenet_dataset_folder_path,
                    object_model_map_dataset_folder_path):
        assert os.path.exists(scannet_object_dataset_folder_path)
        assert os.path.exists(shapenet_dataset_folder_path)
        assert os.path.exists(object_model_map_dataset_folder_path)

        self.reset()

        self.scannet_object_dataset_folder_path = scannet_object_dataset_folder_path
        self.shapenet_dataset_folder_path = shapenet_dataset_folder_path
        self.object_model_map_dataset_folder_path = object_model_map_dataset_folder_path

        self.scene_name_list = os.listdir(
            self.object_model_map_dataset_folder_path)
        return True

    def isSceneValid(self, scene_name):
        return scene_name in self.scene_name_list

    def loadScene(self, scene_name):
        if scene_name == self.scene_name:
            return True

        assert self.isSceneValid(scene_name)

        object_model_map_json_file_path = self.object_model_map_dataset_folder_path + \
            scene_name + "/object_model_map.json"
        assert os.path.exists(object_model_map_json_file_path)

        with open(object_model_map_json_file_path, "r") as f:
            self.scene_object_model_map_dict = json.load(f)
        return True

    def getObjectFileNameList(self, scene_name):
        if not self.isSceneValid(scene_name):
            print("[WARN][ObjectModelMapManager::getObjectFileNameList]")
            print("\t scene not valid!")
            print("\t", scene_name)
            return []

        self.loadScene(scene_name)
        return list(self.scene_object_model_map_dict.keys())

    def isObjectValid(self, scene_name, object_file_name):
        return object_file_name in self.getObjectFileNameList(scene_name)

    def getShapeNetModelDict(self, scene_name, object_file_name):
        self.loadScene(scene_name)

        assert self.scene_object_model_map_dict is not None

        assert self.isObjectValid(scene_name, object_file_name)

        shapenet_model_dict = self.scene_object_model_map_dict[
            object_file_name]

        trans_matrix = np.array(shapenet_model_dict['trans_matrix'])

        t, q, s = decompose_mat4(trans_matrix)
        shapenet_model_dict['translate'] = t.tolist()
        shapenet_model_dict['rotate'] = q.tolist()
        shapenet_model_dict['scale'] = s.tolist()

        trans_matrix_inv = np.linalg.inv(trans_matrix)
        shapenet_model_dict['trans_matrix_inv'] = trans_matrix_inv.tolist()

        t_inv, q_inv, s_inv = decompose_mat4(trans_matrix_inv)
        shapenet_model_dict['translate_inv'] = t_inv.tolist()
        shapenet_model_dict['rotate_inv'] = q_inv.tolist()
        shapenet_model_dict['scale_inv'] = s_inv.tolist()

        scannet_object_file_path = self.scannet_object_dataset_folder_path + \
            scene_name + "/" + object_file_name
        assert os.path.exists(scannet_object_file_path)

        shapenet_model_file_path = self.shapenet_dataset_folder_path + \
            shapenet_model_dict['cad_cat_id'] + "/" + shapenet_model_dict['cad_id'] + \
            "/models/model_normalized.obj"
        assert os.path.exists(shapenet_model_file_path)

        shapenet_model_dict[
            "scannet_object_file_path"] = scannet_object_file_path
        shapenet_model_dict[
            "shapenet_model_file_path"] = shapenet_model_file_path
        return shapenet_model_dict

    def renderScan2CADObjectModelMap(self,
                                     scene_name,
                                     object_file_name,
                                     render_scene=False):
        shapenet_model_dict = self.getShapeNetModelDict(
            scene_name, object_file_name)

        scannet_scene_file_path = None
        if render_scene:
            scannet_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
            scannet_scene_file_path = scannet_dataset_folder_path + \
                scene_name + "/" + scene_name + "_vh_clean.ply"

        renderScan2CADObjectModelMap(shapenet_model_dict,
                                     scannet_scene_file_path)
        return True
