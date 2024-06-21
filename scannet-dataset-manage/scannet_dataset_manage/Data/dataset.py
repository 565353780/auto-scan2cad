#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm

from scannet_dataset_manage.Data.scene import Scene


class Dataset(object):

    def __init__(self, dataset_folder_path):
        self.dataset_folder_path = dataset_folder_path
        if self.dataset_folder_path[-1] != "/":
            self.dataset_folder_path += "/"

        self.scene_dict = {}

        self.update()
        return

    def updateSceneList(self):
        scene_folder_name_list = os.listdir(self.dataset_folder_path)

        print("[INFO][Dataset::updateSceneList]")
        print("\t start load scenes...")
        for scene_folder_name in tqdm(scene_folder_name_list):
            if "scene" not in scene_folder_name:
                continue
            scene_folder_path = self.dataset_folder_path + scene_folder_name + "/"
            self.scene_dict[scene_folder_name] = Scene(scene_folder_path)
        return True

    def update(self):
        assert os.path.exists(self.dataset_folder_path)
        assert self.updateSceneList()
        return True

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[Dataset]")
        print(line_start + "\t dataset_folder_path =",
              self.dataset_folder_path)
        print(line_start + "\t scene_num =", len(self.scene_dict.keys()))
        print(line_start + "\t scene_dict =")
        for _, scene in self.scene_dict.items():
            scene.outputInfo(info_level + 2)
        return True
