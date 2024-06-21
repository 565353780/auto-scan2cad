#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm

from shapenet_dataset_manage.Data.model import Model

from shapenet_dataset_manage.Method.outputs import outputList


class Synset(object):

    def __init__(self, root_path=None, output_info=True):
        self.root_path = None

        self.id = None
        self.model_id_list = []
        self.model_dict = {}

        if root_path is not None:
            self.loadRootPath(root_path, output_info)
        return

    def reset(self):
        self.root_path = None
        self.model_id_list = []
        self.model_dict = {}
        return True

    def loadSynsetId(self):
        self.id = self.root_path.split("/")[-2]
        return True

    def loadModelIdList(self):
        self.model_id_list = os.listdir(self.root_path)
        return True

    def loadModelDict(self, output_info=True):
        if output_info:
            for model_id in tqdm(self.model_id_list):
                model_root_path = self.root_path + model_id + "/"
                model = Model(model_root_path)
                self.model_dict[model_id] = model
            return True

        for model_id in self.model_id_list:
            model_root_path = self.root_path + model_id + "/"
            model = Model(model_root_path)
            self.model_dict[model_id] = model
        return True

    def loadRootPath(self, root_path, output_info=True):
        self.reset()

        if not os.path.exists(root_path):
            print("[ERROR][Synset::loadRootPath]")
            print("\t root_path not exist!")
            return False

        self.root_path = root_path
        if self.root_path[-1] != "/":
            self.root_path += "/"

        if not self.loadSynsetId():
            print("[ERROR][Synset::loadRootPath]")
            print("\t loadSynsetId failed!")
            return False

        if not self.loadModelIdList():
            print("[ERROR][Synset::loadRootPath]")
            print("\t loadModelIdList failed!")
            return False

        if not self.loadModelDict(output_info):
            print("[ERROR][Synset::loadRootPath]")
            print("\t loadModelDict failed!")
            return False
        return True

    def outputInfo(self, info_level=0, print_cols=3):
        line_start = "\t" * info_level
        print(line_start + "[Synset]")
        print(line_start + "\t root_path =", self.root_path)
        print(line_start + "\t id =", self.id)
        print(line_start + "\t model_id_list =")
        outputList(self.model_id_list, info_level + 2, print_cols)
        print(line_start + "\t model size =", len(self.model_id_list))
        return True
