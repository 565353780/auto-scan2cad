#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
from auto_cad_recon.Module.dataset_manager import DatasetManager
from torch.utils.data import Dataset
from tqdm import tqdm

from points_shape_detect.Data.io import IO
from points_shape_detect.Method.matrix import getRotateMatrix
from points_shape_detect.Method.path import createFileFolder, renameFile
from points_shape_detect.Method.trans import (getInverseTrans,
                                              normalizePointArray,
                                              transPointArray)


class CADDataset(Dataset):

    def __init__(self, training=True, training_percent=0.8):
        self.training = training
        self.training_percent = training_percent

        self.cad_model_file_path_list = []
        self.cad_model_bbox_list = []
        self.train_idx_list = []
        self.eval_idx_list = []

        self.loadScan2CAD()
        #  self.loadShapeNet55()
        self.updateIdx()
        return

    def reset(self):
        self.cad_model_file_path_list = []
        return True

    def updateIdx(self, random=False):
        model_num = len(self.cad_model_file_path_list)
        if model_num == 1:
            self.train_idx_list = [0]
            self.eval_idx_list = [0]
            return True

        assert model_num > 0

        train_model_num = int(model_num * self.training_percent)
        if train_model_num == 0:
            train_model_num += 1
        elif train_model_num == model_num:
            train_model_num -= 1

        if random:
            random_idx_list = np.random.choice(np.arange(model_num),
                                               size=model_num,
                                               replace=False)
        else:
            random_idx_list = np.arange(model_num)

        self.train_idx_list = random_idx_list[:train_model_num]
        # FIXME: for test only
        self.train_idx_list = random_idx_list
        self.eval_idx_list = random_idx_list[train_model_num:]
        return True

    def loadScan2CAD(self):
        scannet_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
        scannet_object_dataset_folder_path = "/home/chli/chLi/ScanNet/objects/"
        scannet_bbox_dataset_folder_path = "/home/chli/chLi/ScanNet/bboxes/"
        scan2cad_dataset_folder_path = "/home/chli/chLi/Scan2CAD/scan2cad_dataset/"
        scan2cad_object_model_map_dataset_folder_path = "/home/chli/chLi/Scan2CAD/object_model_maps/"
        shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
        shapenet_udf_dataset_folder_path = "/home/chli/chLi/ShapeNet/udfs/"
        print_progress = True

        dataset_manager = DatasetManager(
            scannet_dataset_folder_path, scannet_object_dataset_folder_path,
            scannet_bbox_dataset_folder_path, scan2cad_dataset_folder_path,
            scan2cad_object_model_map_dataset_folder_path,
            shapenet_dataset_folder_path, shapenet_udf_dataset_folder_path)

        scene_name_list = dataset_manager.getScanNetSceneNameList()

        normal_object_dataset_folder_path = scan2cad_dataset_folder_path + \
            "normal_objects_npy/"

        print("[INFO][CADDataset::loadScan2CAD]")
        print("\t start load scan2cad dataset...")
        for scene_name in tqdm(scene_name_list):
            scene_name = "scene0474_02"

            scene_folder_path = normal_object_dataset_folder_path + scene_name + "/"

            object_file_name_list = dataset_manager.getScanNetObjectFileNameList(
                scene_name)
            for object_file_name in object_file_name_list:
                object_npy_file_path = scene_folder_path + object_file_name.split(
                    ".")[0] + "_obj.npy"
                cad_npy_file_path = scene_folder_path + object_file_name.split(
                    ".")[0] + "_cad.npy"

                if os.path.exists(object_npy_file_path) and os.path.exists(
                        cad_npy_file_path):
                    self.cad_model_file_path_list.append(
                        [object_npy_file_path, cad_npy_file_path])

                    #  self.cad_model_file_path_list.append(cad_npy_file_path)
                    continue

                shapenet_model_dict = dataset_manager.getShapeNetModelDict(
                    scene_name, object_file_name)
                scannet_object_file_path = shapenet_model_dict[
                    'scannet_object_file_path']
                shapenet_model_file_path = shapenet_model_dict[
                    'shapenet_model_file_path']
                trans_matrix_inv = shapenet_model_dict['trans_matrix_inv']

                object_points = IO.get(scannet_object_file_path,
                                       trans_matrix_inv)
                cad_points = IO.get(shapenet_model_file_path)

                tmp_object_npy_file_path = object_npy_file_path[:-4] + "_tmp.npy"
                tmp_cad_npy_file_path = cad_npy_file_path[:-4] + "_tmp.npy"

                createFileFolder(tmp_object_npy_file_path)
                createFileFolder(tmp_cad_npy_file_path)

                np.save(tmp_object_npy_file_path, object_points)
                np.save(tmp_cad_npy_file_path, cad_points)

                renameFile(tmp_object_npy_file_path, object_npy_file_path)
                renameFile(tmp_cad_npy_file_path, cad_npy_file_path)

                self.cad_model_file_path_list.append(
                    [object_npy_file_path, cad_npy_file_path])
                #  self.cad_model_file_path_list.append(cad_npy_file_path)
        return True

    def loadShapeNet55(self):
        dataset_folder_path = '/home/chli/chLi/PoinTr/ShapeNet55/shapenet_pc/'
        file_name_list = os.listdir(dataset_folder_path)
        for file_name in file_name_list:
            self.cad_model_file_path_list.append(dataset_folder_path +
                                                 file_name)
        return True

    def getRotateDataWithPair(self, cad_model_file_path, training=True):
        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        object_npy_file_path, cad_npy_file_path = cad_model_file_path

        object_point_array = IO.get(object_npy_file_path).astype(np.float32)
        cad_point_array = IO.get(cad_npy_file_path).astype(np.float32)

        origin_point_array, origin_cad_point_array = normalizePointArray(
            object_point_array, cad_point_array)

        origin_point_array = torch.from_numpy(origin_point_array).to(
            torch.float32).cuda().unsqueeze(0)
        origin_cad_point_array = torch.from_numpy(origin_cad_point_array).to(
            torch.float32).cuda().unsqueeze(0)

        translate = ((torch.rand(3) - 0.5) * 1000).to(torch.float32).cuda()
        euler_angle = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).cuda()
        #  scale = (1.0 + ((torch.rand(3) - 0.5) * 0.5)).to(torch.float32).cuda()
        scale_value = np.exp((np.random.rand() - 0.5) * 4)
        scale = torch.from_numpy(
            np.array([scale_value, scale_value,
                      scale_value])).to(torch.float32).cuda()
        center = torch.mean(origin_point_array, 0)
        trans_point_array = transPointArray(origin_point_array,
                                            translate,
                                            euler_angle,
                                            scale,
                                            center=center).unsqueeze(0)
        trans_cad_point_array = transPointArray(origin_cad_point_array,
                                                translate,
                                                euler_angle,
                                                scale,
                                                center=center).unsqueeze(0)

        data['inputs']['trans_point_array'] = trans_point_array
        data['inputs']['trans_cad_point_array'] = trans_cad_point_array
        return data

    def getRotateData(self, idx, training=True):
        if self.training:
            idx = self.train_idx_list[idx]
        else:
            idx = self.eval_idx_list[idx]

        cad_model_file_path = self.cad_model_file_path_list[idx]

        if isinstance(cad_model_file_path, list):
            return self.getRotateDataWithPair(cad_model_file_path, training)

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        point_array = IO.get(cad_model_file_path).astype(np.float32)
        origin_point_array = normalizePointArray(point_array)

        origin_point_array = torch.from_numpy(origin_point_array).to(
            torch.float32).cuda()

        translate = ((torch.rand(3) - 0.5) * 1000).to(torch.float32).cuda()
        euler_angle = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32).cuda()
        #  scale = (1.0 + ((torch.rand(3) - 0.5) * 0.1)).to(torch.float32).cuda()
        scale_value = np.exp((np.random.rand() - 0.5) * 4)
        scale = torch.from_numpy([scale_value, scale_value,
                                  scale_value]).to(torch.float32).cuda()
        trans_point_array = transPointArray(origin_point_array, translate,
                                            euler_angle, scale).unsqueeze(0)

        data['inputs']['trans_point_array'] = trans_point_array
        return data

    def getItemWithPair(self, cad_model_file_path, training=True):
        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        object_npy_file_path, cad_npy_file_path = cad_model_file_path

        object_point_array = IO.get(object_npy_file_path).astype(np.float32)
        cad_point_array = IO.get(cad_npy_file_path).astype(np.float32)

        origin_point_array, origin_cad_point_array = normalizePointArray(
            object_point_array, cad_point_array)

        translate = (np.random.rand(3) - 0.5) * 1000
        euler_angle = np.random.rand(3) * 360.0
        #  scale = 1.0 + ((np.random.rand(3) - 0.5) * 0.1)
        scale_value = np.exp((np.random.rand() - 0.5) * 4)
        scale = np.array([scale_value, scale_value, scale_value])

        center = np.mean(origin_point_array, axis=0)

        trans_point_array = transPointArray(origin_point_array,
                                            translate,
                                            euler_angle,
                                            scale,
                                            center=center)
        trans_cad_point_array = transPointArray(origin_cad_point_array,
                                                translate,
                                                euler_angle,
                                                scale,
                                                center=center)

        data['inputs']['trans_point_array'] = torch.from_numpy(
            trans_point_array).float()
        data['inputs']['trans_cad_point_array'] = torch.from_numpy(
            trans_cad_point_array).float()

        if training:
            rotate_matrix = getRotateMatrix(euler_angle)
            data['inputs']['rotate_matrix'] = torch.from_numpy(
                rotate_matrix).to(torch.float32)
        return data

    def getItem(self, cad_model_file_path, training=True):
        if isinstance(cad_model_file_path, list):
            return self.getItemWithPair(cad_model_file_path, training)

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        point_array = IO.get(cad_model_file_path).astype(np.float32)
        origin_point_array = normalizePointArray(point_array)

        translate = (np.random.rand(3) - 0.5) * 1000
        euler_angle = np.random.rand(3) * 360.0
        #  scale = 1.0 + ((np.random.rand(3) - 0.5) * 0.1)
        scale_value = np.exp((np.random.rand() - 0.5) * 4)
        scale = np.array([scale_value, scale_value, scale_value])

        trans_point_array = transPointArray(origin_point_array, translate,
                                            euler_angle, scale)
        data['inputs']['trans_point_array'] = torch.from_numpy(
            trans_point_array).float()

        if training:
            rotate_matrix = getRotateMatrix(euler_angle)
            data['inputs']['rotate_matrix'] = torch.from_numpy(
                rotate_matrix).to(torch.float32)
        return data

    def __getitem__(self, idx, training=True):
        if self.training:
            idx = self.train_idx_list[idx]
        else:
            idx = self.eval_idx_list[idx]

        cad_model_file_path = self.cad_model_file_path_list[idx]

        return self.getItem(cad_model_file_path, training)

    def __len__(self):
        if self.training:
            return len(self.train_idx_list)
        else:
            return len(self.eval_idx_list)
