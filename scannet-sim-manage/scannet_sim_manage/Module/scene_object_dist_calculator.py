#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

import numpy as np
import open3d as o3d
from mesh_manage.Module.channel_mesh import ChannelMesh
from tqdm import tqdm

from scannet_sim_manage.Data.bbox import BBox
from scannet_sim_manage.Data.point import Point

load_method_list = ['channelmesh', 'open3d']
load_method = 'open3d'


class SceneObjectDistCalculator(object):

    def __init__(self,
                 object_folder_path=None,
                 bbox_json_file_path=None,
                 dist_error_max=0.05,
                 print_progress=False):
        self.dist_error_max = dist_error_max

        self.channel_mesh_dict = {}
        self.bbox_dict = {}

        self.open3d_mesh_dict = {}

        if object_folder_path is not None:
            self.loadSceneObject(object_folder_path, print_progress)
        if bbox_json_file_path is not None:
            self.loadObjectBBox(bbox_json_file_path)
        return

    def reset(self):
        del self.channel_mesh_dict
        self.channel_mesh_dict = {}
        self.bbox_dict = {}

        del self.open3d_mesh_dict
        self.open3d_mesh_dict = {}
        return True

    def loadSceneObjectByChannelMesh(self,
                                     object_folder_path,
                                     print_progress=False,
                                     valid_object_file_name_list=None):
        assert os.path.exists(object_folder_path)
        object_file_name_list = os.listdir(object_folder_path)
        for_data = object_file_name_list
        if print_progress:
            print(
                "[INFO][SceneObjectDistCalculator::loadSceneObjectByChannelMesh]"
            )
            print("\t start load scene objects...")
            for_data = tqdm(for_data)
        for object_file_name in for_data:
            if valid_object_file_name_list is not None:
                if object_file_name not in valid_object_file_name_list:
                    continue

            object_file_path = object_folder_path + object_file_name
            self.channel_mesh_dict[object_file_name] = ChannelMesh(
                object_file_path)
        return True

    def loadSceneObjectByOpen3D(self,
                                object_folder_path,
                                print_progress=False,
                                valid_object_file_name_list=None):
        assert os.path.exists(object_folder_path)
        object_file_name_list = os.listdir(object_folder_path)
        for_data = object_file_name_list
        if print_progress:
            print("[INFO][SceneObjectDistCalculator::loadSceneObjectByOpen3D]")
            print("\t start load scene objects...")
            for_data = tqdm(for_data)
        for object_file_name in for_data:
            if valid_object_file_name_list is not None:
                if object_file_name not in valid_object_file_name_list:
                    continue

            object_file_path = object_folder_path + object_file_name
            self.open3d_mesh_dict[object_file_name] = o3d.io.read_point_cloud(
                object_file_path)
        return True

    def loadSceneObject(self,
                        object_folder_path,
                        print_progress=False,
                        valid_object_file_name_list=None):
        assert load_method in load_method_list

        if load_method == 'channelmesh':
            return self.loadSceneObjectByChannelMesh(
                object_folder_path, print_progress,
                valid_object_file_name_list)

        if load_method == 'open3d':
            return self.loadSceneObjectByOpen3D(object_folder_path,
                                                print_progress,
                                                valid_object_file_name_list)

    def loadObjectBBox(self,
                       bbox_json_file_path,
                       valid_object_file_name_list=None):
        assert os.path.exists(bbox_json_file_path)

        with open(bbox_json_file_path, "r") as f:
            data = f.read()
            bbox_json = json.loads(data)

        self.bbox_dict = {}
        for object_file_name, bbox_list in bbox_json.items():
            if valid_object_file_name_list is not None:
                if object_file_name not in valid_object_file_name_list:
                    continue

            self.bbox_dict[object_file_name] = BBox.fromList(bbox_list)
        return True

    def generateBBoxLabelByChannelMesh(self,
                                       point_image,
                                       print_progress=False):
        for_data = point_image.point_array
        if print_progress:
            print(
                "[INFO][SceneObjectDistCalculator::generateBBoxLabelByChannelMesh]"
            )
            print("\t start add bbox label...")
            for_data = tqdm(for_data)
        for i, [x, y, z] in enumerate(for_data):
            point = Point(x, y, z)
            for object_file_name, bbox in self.bbox_dict.items():
                if bbox.isInBBox(point):
                    point_image.addLabel(i, object_file_name, "bbox")
        return point_image

    def generateObjectLabelByChannelMesh(self,
                                         point_image,
                                         print_progress=False):
        point_image = self.generateBBoxLabelByChannelMesh(
            point_image, print_progress)

        for_data = point_image.point_array
        if print_progress:
            print(
                "[INFO][SceneObjectDistCalculator::generateObjectLabelByChannelMesh]"
            )
            print("\t start add object label...")
            for_data = tqdm(for_data)
        for i, [x, y, z] in enumerate(for_data):
            for object_file_name in self.bbox_dict.keys():
                if object_file_name not in point_image.label_dict_list[i].keys(
                ):
                    continue

                dist = self.channel_mesh_dict[object_file_name].getNearestDist(
                    x, y, z)

                if dist <= self.dist_error_max:
                    point_image.addLabel(i, object_file_name, "object")
        return point_image

    def generateObjectLabelByOpen3D(self, point_image, print_progress=False):
        point_idx = np.where(point_image.point_array[:, 0] != float("inf"))[0]

        points = point_image.point_array[point_idx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        object_label_list = list(self.bbox_dict.keys())
        scan_to_object_dist_matrix = []

        for_data = self.bbox_dict.keys()
        if print_progress:
            print(
                "[INFO][SceneObjectDistCalculator::generateObjectLabelByOpen3D]"
            )
            print("\t start calculate object dists...")
            for_data = tqdm(for_data)
        for object_file_name in for_data:
            dist_array = np.array(
                pcd.compute_point_cloud_distance(
                    self.open3d_mesh_dict[object_file_name]))
            scan_to_object_dist_matrix.append(dist_array)

        scan_to_object_dist_matrix = np.dstack(scan_to_object_dist_matrix)[0]
        min_dist_object_idx = np.argmin(scan_to_object_dist_matrix, axis=1)
        min_dist_array = np.min(scan_to_object_dist_matrix, axis=1)

        valid_label_idx = np.where(min_dist_array <= self.dist_error_max)[0]

        for_data = valid_label_idx
        if print_progress:
            print(
                "[INFO][SceneObjectDistCalculator::generateObjectLabelByOpen3D]"
            )
            print("\t start add object labels...")
            for_data = tqdm(for_data)
        for i in valid_label_idx:
            point_i = point_idx[i]
            object_idx = min_dist_object_idx[i]
            point_image.addLabel(point_i, object_label_list[object_idx],
                                 "object")
        return point_image

    def generateObjectLabel(self, point_image, print_progress=False):
        assert load_method in load_method_list

        if load_method == 'channelmesh':
            return self.generateObjectLabelByChannelMesh(
                point_image, print_progress)

        if load_method == 'open3d':
            return self.generateObjectLabelByOpen3D(point_image,
                                                    print_progress)

    def generateBackgroundLabel(self, point_image):
        point_idx = np.where(point_image.point_array[:, 0] != float("inf"))[0]

        for i in point_idx:
            if "object" in point_image.label_dict_list[i].values():
                continue

            point_image.addLabel(i, "background")
        return point_image

    def getLabeledPointImage(self, point_image, print_progress=False):
        point_image = self.generateObjectLabel(point_image, print_progress)

        point_image = self.generateBackgroundLabel(point_image)

        point_image.updateAllLabelBBox2D()
        return point_image
