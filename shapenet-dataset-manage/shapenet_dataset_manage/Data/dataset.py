#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm

from shapenet_dataset_manage.Data.synset import Synset

from shapenet_dataset_manage.Method.outputs import outputList


class Dataset(object):

    def __init__(self, root_path=None, output_info=True):
        self.root_path = None

        self.synset_id_list = []
        self.synset_dict = {}

        if root_path is not None:
            self.loadRootPath(root_path, output_info)
        return

    def reset(self):
        self.root_path = None

        self.synset_id_list = []
        self.synset_dict = {}
        return True

    def loadSynsetIdList(self):
        self.synset_id_list = []
        root_folder_name_list = os.listdir(self.root_path)
        for root_folder_name in root_folder_name_list:
            if not os.path.isdir(self.root_path + root_folder_name):
                continue
            self.synset_id_list.append(root_folder_name)
        return True

    def loadSynsetDict(self, output_info=True):
        if output_info:
            for synset_id in tqdm(self.synset_id_list):
                synset_root_path = self.root_path + synset_id + "/"
                synset = Synset(synset_root_path, False)
                self.synset_dict[synset_id] = synset
            return True

        for synset_id in self.synset_id_list:
            synset_root_path = self.root_path + synset_id + "/"
            synset = Synset(synset_root_path, False)
            self.synset_dict[synset_id] = synset
        return True

    def loadRootPath(self, root_path, output_info=True):
        self.reset()

        assert os.path.exists(root_path)

        self.root_path = root_path

        assert self.loadSynsetIdList()

        assert self.loadSynsetDict(output_info)
        return True

    def outputInfo(self, info_level=0, print_cols=10):
        line_start = "\t" * info_level
        print(line_start + "[Dataset]")
        print(line_start + "\t root_path =", self.root_path)
        print(line_start + "\t synset_id_list =")
        outputList(self.synset_id_list, info_level + 2, print_cols)
        print(line_start + "\t synset size =", len(self.synset_id_list))
        return True
