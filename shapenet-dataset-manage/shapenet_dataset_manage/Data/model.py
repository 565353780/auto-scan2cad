#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json

from shapenet_dataset_manage.Method.outputs import outputJson


class Model(object):

    def __init__(self, root_path=None):
        self.root_path = None

        self.id = None
        self.normalized_json = None
        self.normalized_obj_file_path = None
        self.normalized_mtl_file_path = None
        self.normalized_solid_binvox_file_path = None
        self.normalized_surface_binvox_file_path = None

        self.exist_images = False
        self.images_folder_path = None

        self.exist_screenshots = False
        self.screenshots_folder_path = None

        if root_path is not None:
            self.loadRootPath(root_path)
        return

    def reset(self):
        self.root_path = None
        self.id = None
        self.normalized_json = None
        self.normalized_obj_file_path = None
        self.normalized_mtl_file_path = None
        self.normalized_solid_binvox_file_path = None
        self.normalized_surface_binvox_file_path = None

        self.exist_images = False
        self.images_folder_path = None

        self.exist_screenshots = False
        self.screenshots_folder_path = None
        return True

    def loadModelsFolder(self):
        model_files_basepath = self.root_path + "models/model_normalized."

        self.id = self.root_path.split("/")[-2]

        json_file_path = model_files_basepath + "json"
        if os.path.exists(json_file_path):
            with open(json_file_path, "r") as f:
                self.normalized_json = json.load(f)
        else:
            print("[WARN][Model::loadModelsFolder]")
            print("\t model[" + self.id + "] json file not exist!")

        self.normalized_obj_file_path = model_files_basepath + "obj"
        if not os.path.exists(self.normalized_obj_file_path):
            self.normalized_obj_file_path = None

        self.normalized_mtl_file_path = model_files_basepath + "mtl"
        if not os.path.exists(self.normalized_mtl_file_path):
            self.normalized_mtl_file_path = None

        self.normalized_solid_binvox_file_path = model_files_basepath + "solid.binvox"
        if not os.path.exists(self.normalized_solid_binvox_file_path):
            self.normalized_solid_binvox_file_path = None

        self.normalized_surface_binvox_file_path = model_files_basepath + "surface.binvox"
        if not os.path.exists(self.normalized_surface_binvox_file_path):
            self.normalized_surface_binvox_file_path = None
        return True

    def loadImagesFolder(self):
        images_folder_path = self.root_path + "images/"
        if os.path.exists(images_folder_path):
            self.exist_images = True
            self.images_folder_path = images_folder_path
        return True

    def loadScreenShotsFolder(self):
        screenshots_folder_path = self.root_path + "screenshots/"
        if os.path.exists(screenshots_folder_path):
            self.exist_screenshots = True
            self.screenshots_folder_path = screenshots_folder_path
        return True

    def loadRootPath(self, root_path):
        self.reset()

        if not os.path.exists(root_path):
            print("[ERROR][Model::loadRootPath]")
            print("\t root_path not exist!")
            return False

        self.root_path = root_path
        if self.root_path[-1] != "/":
            self.root_path += "/"

        if not self.loadModelsFolder():
            print("[ERROR][Model::loadRootPath]")
            print("\t loadModelsFolder failed!")
            return False

        if not self.loadImagesFolder():
            print("[ERROR][Model::loadRootPath]")
            print("\t loadImagesFolder failed!")
            return False

        if not self.loadScreenShotsFolder():
            print("[ERROR][Model::loadRootPath]")
            print("\t loadScreenShotsFolder failed!")
            return False
        return True

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[Model]")
        print(line_start + "\t root_path =", self.root_path)
        print(line_start + "\t normalized_json =")
        outputJson(self.normalized_json, info_level + 2)
        print(line_start + "\t normalized_obj_file_path =",
              self.normalized_obj_file_path)
        print(line_start + "\t normalized_mtl_file_path =",
              self.normalized_mtl_file_path)
        print(line_start + "\t normalized_solid_binvox_file_path =",
              self.normalized_solid_binvox_file_path)
        print(line_start + "\t normalized_surface_binvox_file_path =",
              self.normalized_surface_binvox_file_path)
        if self.exist_images:
            print(line_start + "\t images_folder_path =",
                  self.images_folder_path)
        if self.exist_screenshots:
            print(line_start + "\t screenshots_folder_path =",
                  self.screenshots_folder_path)
        return True
