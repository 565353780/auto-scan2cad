#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import torch
import numpy as np
import open3d as o3d
from copy import deepcopy

from auto_cad_recon.Method.bbox import getOBBFromABB
from auto_cad_recon.Module.dataset_manager import DatasetManager
from points_shape_detect.Data.bbox import BBox
from points_shape_detect.Loss.ious import IoULoss
from points_shape_detect.Method.bbox import (getBBoxPointList,
                                             getOpen3DBBoxFromBBoxArray)
from points_shape_detect.Method.matrix import getRotateMatrix
from points_shape_detect.Method.trans import (getInverseTrans,
                                              normalizePointArray,
                                              transPointArray)
from scene_layout_detect.Module.layout_map_builder import LayoutMapBuilder
from torch.utils.data import Dataset
from tqdm import tqdm

from global_pose_refine.Data.obb import OBB
from global_pose_refine.Method.path import createFileFolder, renameFile
from global_pose_refine.Module.relation_calculator import RelationCalculator


class ObjectPositionDataset(Dataset):
    def __init__(self, training=True, training_percent=0.8):
        pose_weight = 0.5
        self.relation_calculator = RelationCalculator(pose_weight)

        self.training = training
        self.training_percent = training_percent

        self.object_position_set_list = []
        self.train_idx_list = []
        self.eval_idx_list = []

        self.loadScan2CAD()
        self.updateIdx()

        self.repeat_time = 1
        return

    def reset(self):
        self.object_position_set_list = []
        return True

    def updateIdx(self, random=False):
        loaded_data_num = len(self.object_position_set_list)
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

        dataset_folder_path = scan2cad_dataset_folder_path + "object_position_dataset/"

        print("[INFO][ObjectPositionDataset::loadScan2CAD]")
        print("\t start load scan2cad dataset...")
        for scene_name in tqdm(scene_name_list):
            #  scene_name = "scene0474_02"

            scene_folder_path = dataset_folder_path + scene_name + "/"

            object_obb_file_path = scene_folder_path + "object_obb.npy"

            if os.path.exists(object_obb_file_path):
                object_obb = np.load(object_obb_file_path)
            else:
                object_file_name_list = dataset_manager.getScanNetObjectFileNameList(
                    scene_name)

                object_obb_list = []
                for object_file_name in object_file_name_list:
                    shapenet_model_dict = dataset_manager.getShapeNetModelDict(
                        scene_name, object_file_name)
                    trans_matrix = shapenet_model_dict['trans_matrix']
                    shapenet_model_file_path = shapenet_model_dict[
                        'shapenet_model_file_path']

                    #  object_abb = np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5],
                    #  dtype=float)

                    cad_mesh = o3d.io.read_triangle_mesh(
                        shapenet_model_file_path)
                    cad_bbox = cad_mesh.get_axis_aligned_bounding_box()
                    min_point = cad_bbox.min_bound
                    max_point = cad_bbox.max_bound
                    object_abb = np.hstack((min_point, max_point))

                    object_obb_points = getOBBFromABB(object_abb)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(object_obb_points)
                    pcd.transform(trans_matrix)
                    object_obb = np.array(pcd.points)

                    object_obb_list.append(object_obb)

                object_obb = np.array(object_obb_list)

                tmp_object_obb_file_path = object_obb_file_path[:-4] + "_tmp.npy"

                createFileFolder(tmp_object_obb_file_path)

                np.save(tmp_object_obb_file_path, object_obb)

                renameFile(tmp_object_obb_file_path, object_obb_file_path)

            object_position_set = [object_obb]
            self.object_position_set_list.append(object_position_set)
        return True

    def getItem(self, idx, random_object_num=None):
        wall_height = 3
        convex_hull_scale = 1e8

        idx = int(idx / self.repeat_time)

        if self.training:
            idx = self.train_idx_list[idx]
        else:
            idx = self.eval_idx_list[idx]

        object_position_set = self.object_position_set_list[idx]
        object_obb = object_position_set[0]

        if random_object_num is None:
            random_object_num = np.random.randint(1, object_obb.shape[0] + 1)
        random_idx = np.random.choice(np.arange(random_object_num),
                                      random_object_num,
                                      replace=False)

        object_obb = object_obb[random_idx]

        layout_map_builder = LayoutMapBuilder()

        points = deepcopy(object_obb[:, :, :2]).reshape(-1, 2)
        scale_points = (deepcopy(points) * convex_hull_scale).astype(int)
        scale_hull = cv2.convexHull(scale_points)
        hull = scale_hull.reshape(-1, 2).astype(float) / convex_hull_scale
        hull_points = np.hstack([hull, np.zeros([hull.shape[0], 1])])
        layout_map_builder.addBound(hull_points)

        layout_map_builder.updateLayoutMesh(skip_floor=True)

        floor_position = layout_map_builder.layout_map.floor_array
        floor_normal = np.array([[0.0, 0.0, 1.0]
                                 for _ in range(floor_position.shape[0])])
        floor_z_value = np.array([[0.0]
                                  for _ in range(floor_position.shape[0])])
        floor_abb = np.hstack([floor_position, floor_position
                               ]) + [-1000, -1000, -0.01, 1000, 1000, 0]
        floor_obb = np.array(
            [OBB.fromABBList(abb).toArray() for abb in floor_abb])

        wall_position_list = []
        wall_normal_list = []
        for i in range(floor_position.shape[0]):
            start_idx = i
            end_idx = (i + 1) % floor_position.shape[0]

            wall_position = np.array([
                floor_position[start_idx], floor_position[end_idx],
                floor_position[end_idx] + [0.0, 0.0, wall_height],
                floor_position[start_idx] + [0.0, 0.0, wall_height]
            ])
            wall_diff = wall_position[1] - wall_position[0]
            wall_normal = np.array([-wall_diff[1], wall_diff[0], 0])
            wall_normal = wall_normal / np.linalg.norm(wall_normal)

            wall_position_list.append(wall_position)
            wall_normal_list.append(wall_normal)

        wall_position = np.array(wall_position_list)
        wall_normal = np.array(wall_normal_list)
        wall_obb = np.hstack([wall_position, wall_position])
        wall_obb[:, :4, :] -= wall_normal.reshape(-1, 1, 3) * 0.01
        wall_obb[:, 4:, :] += wall_normal.reshape(-1, 1, 3) * 0.01

        object_num = object_obb.shape[0]
        wall_num = wall_position.shape[0]
        floor_num = floor_position.shape[0]

        object_abb_list = []
        object_obb_center_list = []
        translate_list = []
        euler_angle_list = []
        rotate_matrix_list = []
        scale_list = []
        translate_inv_list = []
        euler_angle_inv_list = []
        rotate_matrix_inv_list = []
        scale_inv_list = []
        trans_object_obb_list = []
        trans_object_abb_list = []
        trans_object_obb_center_list = []
        for obb in object_obb:
            object_abb = np.hstack((np.min(obb, axis=0), np.max(obb, axis=0)))

            object_obb_center = np.mean(obb, axis=0)

            translate = (np.random.rand(3) - 0.5) * 0.5
            euler_angle = (np.random.rand(3) - 0.5) * np.array(
                [90.0, 90.0, 360.0])
            rotate_matrix = getRotateMatrix(euler_angle)
            zero_euler_angle = np.array([0.0, 0.0, 0.0])
            scale = 1.0 + (np.random.rand(3) - 0.5) * 1.0

            translate_inv, euler_angle_inv, scale_inv = getInverseTrans(
                translate, euler_angle, scale)
            rotate_matrix_inv = rotate_matrix.T

            trans_object_obb = transPointArray(obb, translate,
                                               zero_euler_angle, scale)
            trans_object_obb_center = np.mean(trans_object_obb, axis=0)

            trans_object_obb = trans_object_obb - trans_object_obb_center
            trans_object_obb = trans_object_obb @ rotate_matrix
            trans_object_obb = trans_object_obb + trans_object_obb_center

            trans_object_abb = np.hstack(
                (np.min(trans_object_obb,
                        axis=0), np.max(trans_object_obb, axis=0)))

            object_abb_list.append(object_abb)
            object_obb_center_list.append(object_obb_center)
            translate_list.append(translate)
            euler_angle_list.append(euler_angle)
            rotate_matrix_list.append(rotate_matrix)
            scale_list.append(scale)
            translate_inv_list.append(translate_inv)
            euler_angle_inv_list.append(euler_angle_inv)
            rotate_matrix_inv_list.append(rotate_matrix_inv)
            scale_inv_list.append(scale_inv)
            trans_object_obb_list.append(trans_object_obb)
            trans_object_abb_list.append(trans_object_abb)
            trans_object_obb_center_list.append(trans_object_obb_center)

        object_abb = np.array(object_abb_list)
        object_obb_center = np.array(object_obb_center_list)
        translate = np.array(translate_list)
        euler_angle = np.array(euler_angle_list)
        rotate_matrix = np.array(rotate_matrix_list)
        scale = np.array(scale_list)
        translate_inv = np.array(translate_inv_list)
        euler_angle_inv = np.array(euler_angle_inv_list)
        rotate_matrix_inv = np.array(rotate_matrix_inv_list)
        scale_inv = np.array(scale_inv_list)
        trans_object_obb = np.array(trans_object_obb_list)
        trans_object_abb = np.array(trans_object_abb_list)
        trans_object_obb_center = np.array(trans_object_obb_center_list)

        trans_obb_list = np.vstack([trans_object_obb, wall_obb, floor_obb])
        valid_idx_list = range(len(trans_object_obb))
        relation_matrix = self.relation_calculator.calculateRelationsByOBBValueList(
            trans_obb_list, valid_idx_list)

        trans_abb_list = np.array(
            [OBB(trans_obb).toABBArray() for trans_obb in trans_obb_list])
        trans_obb_center_list = np.array([
            np.mean(OBB(trans_obb).points, axis=0)
            for trans_obb in trans_obb_list
        ])

        trans_obb_center_dist_list = [
            np.linalg.norm(center2 - center1, ord=2)
            for center1 in trans_obb_center_list
            for center2 in trans_obb_center_list
        ]

        floor_position = torch.from_numpy(floor_position).float().reshape(
            floor_num, -1)
        floor_normal = torch.from_numpy(floor_normal).float()
        floor_z_value = torch.from_numpy(floor_z_value).float()

        wall_position = torch.from_numpy(wall_position).float().reshape(
            wall_num, -1)
        wall_normal = torch.from_numpy(wall_normal).float()

        object_obb = torch.from_numpy(object_obb).float().reshape(
            object_num, -1)
        object_abb = torch.from_numpy(object_abb).float()
        object_obb_center = torch.from_numpy(object_obb_center).float()

        translate = torch.from_numpy(translate).float()
        euler_angle = torch.from_numpy(euler_angle).float()
        rotate_matrix = torch.from_numpy(rotate_matrix).float()
        scale = torch.from_numpy(scale).float()
        translate_inv = torch.from_numpy(translate_inv).float()
        euler_angle_inv = torch.from_numpy(euler_angle_inv).float()
        rotate_matrix_inv = torch.from_numpy(rotate_matrix_inv).float()
        scale_inv = torch.from_numpy(scale_inv).float()

        trans_object_obb = torch.from_numpy(trans_object_obb).float().reshape(
            object_num, -1)
        trans_object_abb = torch.from_numpy(trans_object_abb).float()
        trans_object_obb_center = torch.from_numpy(
            trans_object_obb_center).float()

        trans_abb_list = torch.from_numpy(trans_abb_list).float().reshape(
            -1, 6)

        trans_abb_eiou_list = [
            IoULoss.EIoU(bbox1, bbox2) for bbox1 in trans_abb_list
            for bbox2 in trans_abb_list
        ]

        trans_obb_center_dist = torch.tensor(
            trans_obb_center_dist_list).float().unsqueeze(-1)
        trans_abb_eiou = torch.tensor(trans_abb_eiou_list).float().unsqueeze(
            -1)

        relation_matrix = torch.from_numpy(relation_matrix).float()

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        data['inputs']['floor_position'] = floor_position
        data['inputs']['floor_normal'] = floor_normal
        data['inputs']['floor_z_value'] = floor_z_value

        data['inputs']['wall_position'] = wall_position
        data['inputs']['wall_normal'] = wall_normal

        data['inputs']['trans_object_obb'] = trans_object_obb
        data['inputs']['trans_object_abb'] = trans_object_abb
        data['inputs']['trans_object_obb_center'] = trans_object_obb_center

        data['inputs']['trans_obb_center_dist'] = trans_obb_center_dist
        data['inputs']['trans_abb_eiou'] = trans_abb_eiou

        data['inputs']['object_obb'] = object_obb
        data['inputs']['object_abb'] = object_abb
        data['inputs']['object_obb_center'] = object_obb_center

        data['inputs']['translate_inv'] = translate_inv
        data['inputs']['rotate_matrix_inv'] = rotate_matrix_inv
        data['inputs']['scale_inv'] = scale_inv

        data['inputs']['relation_matrix'] = relation_matrix
        return data

    def getBatchItem(self, idx):
        batch_size = 10

        idx = int(idx / self.repeat_time)

        if self.training:
            idx = self.train_idx_list[idx]
        else:
            idx = self.eval_idx_list[idx]

        object_position_set = self.object_position_set_list[idx]
        object_obb = object_position_set[0]

        random_object_num = np.random.randint(1, object_obb.shape[0] + 1)

        data_list = []
        for i in range(batch_size):
            data = self.getItem(idx, random_object_num)
            data_list.append(data)

        batch_data = {
            'inputs': {},
            'predictions': {},
            'losses': {},
            'logs': {}
        }

        for key in data_list[0]['inputs'].keys():
            value_list = []
            for i in range(batch_size):
                value_list.append(data_list[i]['inputs'][key].unsqueeze(0))
            batch_value = torch.cat(value_list, 0).cuda()
            batch_data['inputs'][key] = batch_value
        return batch_data

    def __getitem__(self, idx):
        return self.getItem(idx)

    def __len__(self):
        if self.training:
            return len(self.train_idx_list) * self.repeat_time
        else:
            return len(self.eval_idx_list) * self.repeat_time
