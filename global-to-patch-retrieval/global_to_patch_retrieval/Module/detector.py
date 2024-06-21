#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import open3d as o3d
import torch

from global_to_patch_retrieval.Dataset.scan2cad import Scan2CAD
from global_to_patch_retrieval.Method.device import toCpu, toCuda, toNumpy
from global_to_patch_retrieval.Model.retrieval_net import RetrievalNet


class Detector(object):

    def __init__(self, model_file_path=None):
        self.model = RetrievalNet(True).cuda()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        assert os.path.exists(model_file_path)

        print("[INFO][Detector::loadModel]")
        print("\t start loading model from :")
        print("\t", model_file_path)
        model_dict = torch.load(model_file_path)
        self.model.load_state_dict(model_dict['model'])
        return True

    def detectDataset(self):
        dataset_file = \
            "/home/chli/chLi/Scan-CAD Object Similarity Dataset/scan2cad_objects_split.json"
        scannet_folder = \
            "/home/chli/chLi/Scan-CAD Object Similarity Dataset/objects_aligned/"
        shapenet_folder = \
            "/home/chli/chLi/Scan-CAD Object Similarity Dataset/objects_aligned/"

        dataset = Scan2CAD(dataset_file, scannet_folder, shapenet_folder,
                           ["train"])

        for i in range(len(dataset)):
            load_data = dataset.__getitem__(i)
            scan_name = load_data['inputs']['scan_name']
            scannet_object_path = dataset.scannet_root + scan_name + '.sdf'
            print(scan_name)
            print(scannet_object_path)
            assert os.path.exists(scannet_object_path)

            scan_content = load_data['inputs']['scan_content']

            data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

            scan_content_tensor = torch.from_numpy(
                scan_content).cuda().unsqueeze(0)
            data['inputs']['scan_content'] = scan_content_tensor

            data = self.model(data)

            embed_feature = data['predictions']['embed_feature'].detach().cpu(
            ).numpy().reshape(-1)

            print(embed_feature.shape)
            return
        return True
