#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import open3d as o3d
from copy import deepcopy
from torch.utils.data import Dataset

from udf_generate.Method.udfs import getPointUDF


class RetrievalDataset(Dataset):

    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager

        self.scannet_scene_name_list = self.dataset_manager.getScanNetSceneNameList(
        )

        # collect all scene-object udf files as list-dict
        self.scannet_scene_name = None
        self.scannet_scene_object_file_name_list = []
        self.scannet_object_points_dict = {}
        self.shapenet_model_tensor_dict_dict = {}
        self.test_shapenet_model_tensor_dict_dict = {}
        return

    def reset(self):
        self.scannet_scene_name = None
        self.scannet_scene_object_file_name_list = []
        return True

    def loadScene(self, scannet_scene_name):
        assert scannet_scene_name in self.scannet_scene_name_list

        self.reset()

        self.scannet_scene_name = scannet_scene_name
        self.scannet_scene_object_file_name_list = self.dataset_manager.getScanNetObjectFileNameList(
            self.scannet_scene_name)

        for scannet_scene_object_file_name in self.scannet_scene_object_file_name_list:
            shapenet_model_tensor_dict = self.dataset_manager.getShapeNetModelTensorDict(
                self.scannet_scene_name, scannet_scene_object_file_name, False)
            self.shapenet_model_tensor_dict_dict[
                scannet_scene_object_file_name] = shapenet_model_tensor_dict

            test_shapenet_model_tensor_dict = self.dataset_manager.getShapeNetModelTensorDict(
                self.scannet_scene_name, scannet_scene_object_file_name)
            self.test_shapenet_model_tensor_dict_dict[
                scannet_scene_object_file_name] = test_shapenet_model_tensor_dict

            scannet_object_file_path = shapenet_model_tensor_dict[
                'scannet_object_file_path']

            points = np.array(
                o3d.io.read_point_cloud(scannet_object_file_path).points)
            self.scannet_object_points_dict[
                scannet_scene_object_file_name] = points
        return True

    def loadSceneByIdx(self, scannet_scene_name_idx):
        assert scannet_scene_name_idx <= len(self.scannet_scene_name_list)

        return self.loadScene(
            self.scannet_scene_name_list[scannet_scene_name_idx])

    def __getitem__(self, index, test=False):
        assert self.scannet_scene_name is not None

        assert index <= len(self.scannet_scene_object_file_name_list)

        scannet_scene_object_file_name = self.scannet_scene_object_file_name_list[
            index]

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}
        data['inputs']['scannet_scene_name'] = self.scannet_scene_name
        data['inputs'][
            'scannet_object_file_name'] = scannet_scene_object_file_name
        if test:
            data['inputs'].update(self.test_shapenet_model_tensor_dict_dict[
                scannet_scene_object_file_name])
        else:
            data['inputs'].update(self.shapenet_model_tensor_dict_dict[
                scannet_scene_object_file_name])

        points = self.scannet_object_points_dict[
            scannet_scene_object_file_name]

        point_num = points.shape[0]
        sample_percent = 0.5
        sample_scale = sample_percent + (
            1.0 - sample_percent) * random.randint(0, 100) / 100.0
        sample_num = int(point_num * sample_scale)

        random_idx_array = np.array(
            np.random.choice(np.arange(point_num),
                             size=sample_num,
                             replace=False))

        merged_point_array = deepcopy(points)[random_idx_array]

        min_point_list = [np.min(merged_point_array[:, i]) for i in range(3)]
        max_point_list = [np.max(merged_point_array[:, i]) for i in range(3)]

        data['predictions']['init_translate'] = torch.tensor([
            -(min_point_list[i] + max_point_list[i]) for i in range(3)
        ]).type(torch.FloatTensor)

        if test:
            data['predictions']['init_translate'] = data['predictions'][
                'init_translate'].reshape(1, 3)

        diff_max = np.max(
            [max_point_list[i] - min_point_list[i] for i in range(3)])

        if diff_max > 0:
            data['predictions']['init_scale'] = torch.tensor(
                [1.0 / diff_max for _ in range(3)]).type(torch.FloatTensor)
        else:
            data['predictions']['init_scale'] = torch.tensor(
                [1.0 for _ in range(3)]).type(torch.FloatTensor)

        if test:
            data['predictions']['init_scale'] = data['predictions'][
                'init_scale'].reshape(1, 3)

        point_udf = getPointUDF(merged_point_array)

        data['predictions']['point_udf'] = torch.tensor(point_udf).type(
            torch.FloatTensor)

        if test:
            data['predictions']['point_udf'] = data['predictions'][
                'point_udf'].reshape(1, 32, 32, 32)
        return data

    def __len__(self):
        return len(self.scannet_scene_object_file_name_list)
