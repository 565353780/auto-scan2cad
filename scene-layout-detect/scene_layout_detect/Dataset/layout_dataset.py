#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import open3d as o3d
import torch
from auto_cad_recon.Module.dataset_manager import DatasetManager
from points_shape_detect.Data.bbox import BBox
from points_shape_detect.Loss.ious import IoULoss
from points_shape_detect.Method.bbox import (getBBoxPointList,
                                             getOpen3DBBoxFromBBoxArray)
from torch.utils.data import Dataset
from tqdm import tqdm

from scannet_layout_dataset_manage.Module.dataset_loader import DatasetLoader

from scene_layout_detect.Method.path import createFileFolder, renameFile


class LayoutDataset(Dataset):

    def __init__(self, training=True, training_percent=0.8):
        self.training = training
        self.training_percent = training_percent

        self.layout_set_list = []
        self.train_idx_list = []
        self.eval_idx_list = []

        self.loadScanNetLayout()
        self.updateIdx()

        self.repeat_time = 1
        return

    def reset(self):
        self.layout_set_list = []
        return True

    def updateIdx(self, random=False):
        loaded_data_num = len(self.layout_set_list)
        if loaded_data_num == 1:
            self.train_idx_list = [0]
            self.eval_idx_list = [0]
            return True

        assert loaded_data_num > 0

        train_data_num = int(loaded_data_num * self.training_percent)
        if train_data_num == 0:
            train_data_num += 1
        elif train_data_num == loaded_data_num:
            train_data_num -= 1

        if random:
            random_idx_list = np.random.choice(np.arange(loaded_data_num),
                                               size=loaded_data_num,
                                               replace=False)
        else:
            random_idx_list = np.arange(loaded_data_num)

        self.train_idx_list = random_idx_list[:train_data_num]
        self.eval_idx_list = random_idx_list[train_data_num:]
        return True

    def loadScanNetLayout(self):
        scannet_layout_dataset_folder_path = "/home/chli/chLi/ScanNetLayout/"
        print_progress = True

        dataset_loader = DatasetLoader(scannet_layout_dataset_folder_path)

        scene_name_list = dataset_loader.getSceneNameList()

        dataset_folder_path = scan2cad_dataset_folder_path + "object_position_dataset/"

        print("[INFO][LayoutDataset::loadScanNetLayout]")
        print("\t start load scannet layout dataset...")
        for scene_name in tqdm(scene_name_list):
            view_name_list = dataset_loader.getViewNameList(scene_name)

            for view_name in view_name_list:
                depth_image, layout_depth_image = dataset_loader.getUniformDepthImages(
                    scene_name, view_name)
            # TODO: finish this func

            scene_folder_path = dataset_folder_path + scene_name + "/"
            bbox_array_file_path = scene_folder_path + "bbox_array.npy"
            center_array_file_path = scene_folder_path + "center_array.npy"

            if os.path.exists(bbox_array_file_path) and os.path.exists(
                    center_array_file_path):
                bbox_array = np.load(bbox_array_file_path)
                center_array = np.load(center_array_file_path)
            else:
                object_file_name_list = dataset_manager.getScanNetObjectFileNameList(
                    scene_name)

                bbox_list = []
                center_list = []
                for object_file_name in object_file_name_list:
                    shapenet_model_dict = dataset_manager.getShapeNetModelDict(
                        scene_name, object_file_name)
                    trans_matrix = shapenet_model_dict['trans_matrix']
                    shapenet_model_file_path = shapenet_model_dict[
                        'shapenet_model_file_path']

                    cad_mesh = o3d.io.read_triangle_mesh(
                        shapenet_model_file_path)
                    cad_mesh.transform(trans_matrix)
                    cad_bbox = cad_mesh.get_axis_aligned_bounding_box()

                    cad_center = cad_bbox.get_center()
                    min_point = cad_bbox.min_bound
                    max_point = cad_bbox.max_bound
                    bbox = np.hstack((min_point, max_point))

                    bbox_list.append(bbox)
                    center_list.append(cad_center)

                bbox_array = np.array(bbox_list)
                center_array = np.array(center_list)

                tmp_bbox_array_file_path = bbox_array_file_path[:-4] + "_tmp.npy"
                tmp_center_array_file_path = bbox_center_file_path[:-4] + "_tmp.npy"

                createFileFolder(tmp_bbox_array_file_path)
                createFileFolder(tmp_center_array_file_path)

                np.save(tmp_bbox_array_file_path, bbox_array)
                np.save(tmp_center_array_file_path, center_array)

                renameFile(tmp_bbox_array_file_path, bbox_array_file_path)
                renameFile(tmp_center_array_file_path, center_array_file_path)

            object_position_set = [bbox_array, center_array]
            self.object_position_set_list.append(object_position_set)
        return True

    def __getitem__(self, idx, training=True):
        idx = int(idx / self.repeat_time)

        if self.training:
            idx = self.train_idx_list[idx]
        else:
            idx = self.eval_idx_list[idx]

        object_position_set = self.object_position_set_list[idx]
        bbox_array, center_array = object_position_set

        random_object_num = np.random.randint(1, bbox_array.shape[0] + 1)
        random_idx = np.random.choice(np.arange(random_object_num),
                                      random_object_num,
                                      replace=False)

        bbox_array = bbox_array[random_idx]
        center_array = center_array[random_idx]

        bbox_array = torch.from_numpy(bbox_array).float()
        center_array = torch.from_numpy(center_array).float()

        layout_bbox_array = torch.zeros(5, 6).float()
        layout_center_array = torch.zeros(5, 3).float()

        object_num = bbox_array.shape[0]
        layout_num = layout_bbox_array.shape[0]

        random_bbox_noise = (torch.rand(object_num, 6) - 0.5) * 0.5
        random_center_noise = (torch.rand(object_num, 3) - 0.5) * 0.5

        random_bbox_array = bbox_array + random_bbox_noise
        random_center_array = center_array + random_center_noise

        center_dist_list = [
            np.linalg.norm(center2 - center1, ord=2)
            for center1 in random_center_array
            for center2 in random_center_array
        ]
        bbox_eiou_list = [
            IoULoss.EIoU(bbox1, bbox2) for bbox1 in random_bbox_array
            for bbox2 in random_bbox_array
        ]
        center_dist = torch.tensor(center_dist_list).float().unsqueeze(-1)
        bbox_eiou = torch.tensor(bbox_eiou_list).float().unsqueeze(-1)

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        data['inputs']['object_bbox'] = random_bbox_array
        data['inputs']['object_center'] = random_center_array
        data['inputs']['layout_bbox'] = layout_bbox_array
        data['inputs']['layout_center'] = layout_center_array
        data['inputs']['center_dist'] = center_dist
        data['inputs']['bbox_eiou'] = bbox_eiou

        #  if self.training:
        data['inputs']['gt_object_bbox'] = bbox_array
        data['inputs']['gt_object_center'] = center_array
        data['inputs']['gt_layout_bbox'] = layout_bbox_array
        data['inputs']['gt_layout_center'] = layout_center_array
        return data

    def __len__(self):
        if self.training:
            return len(self.train_idx_list) * self.repeat_time
        else:
            return len(self.eval_idx_list) * self.repeat_time
