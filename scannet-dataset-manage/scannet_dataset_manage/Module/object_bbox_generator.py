#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from tqdm import tqdm

from mesh_manage.Module.channel_mesh import ChannelMesh

from scannet_dataset_manage.Method.path import createFileFolder, renameFile


class ObjectBBoxGenerator(object):

    def __init__(self, objects_folder_path, save_json_folder_path):
        self.objects_folder_path = objects_folder_path
        self.save_json_folder_path = save_json_folder_path
        return

    def getObjectBBox(self, object_file_path):
        assert os.path.exists(object_file_path)
        return ChannelMesh(object_file_path).getBBox().toList()

    def getSceneObjectBBoxDict(self, scene_folder_path):
        assert os.path.exists(scene_folder_path)

        scene_object_bbox_dict = {}

        object_file_name_list = os.listdir(scene_folder_path)
        print("[INFO][ObjectBBoxGenerator::getSceneObjectBBoxDict]")
        print("\t start get scene object bbox dict...")
        for object_file_name in tqdm(object_file_name_list):
            object_file_path = scene_folder_path + object_file_name

            scene_object_bbox_dict[object_file_name] = self.getObjectBBox(
                object_file_path)
        return scene_object_bbox_dict

    def generateObjectBBoxJson(self):
        assert os.path.exists(self.objects_folder_path)

        scene_name_list = os.listdir(self.objects_folder_path)
        for i, scene_name in enumerate(scene_name_list):
            save_json_file_path = self.save_json_folder_path + scene_name + "/" + \
                "object_bbox.json"
            if os.path.exists(save_json_file_path):
                continue

            scene_folder_path = self.objects_folder_path + scene_name + "/"

            print("[INFO][ObjectBBoxGenerator::loadAllObjectBBox]")
            print("\t start split scene", scene_name, ",", i + 1, "/",
                  len(scene_name_list), "...")
            scene_object_bbox_dict = self.getSceneObjectBBoxDict(
                scene_folder_path)

            tmp_save_json_file_path = save_json_file_path[:-5] + "_tmp.json"
            createFileFolder(tmp_save_json_file_path)

            with open(tmp_save_json_file_path, "w") as f:
                f.write(json.dumps(scene_object_bbox_dict, indent=4))

            renameFile(tmp_save_json_file_path, save_json_file_path)
        return True
