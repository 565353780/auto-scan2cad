#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

import numpy as np
import open3d as o3d
from tqdm import tqdm

from scan2cad_dataset_manage.Data.point import Point
from scan2cad_dataset_manage.Data.bbox import BBox
from scan2cad_dataset_manage.Method.path import (createFileFolder, removeFile,
                                                 renameFile)
from scan2cad_dataset_manage.Module.dataset_loader import DatasetLoader


class ObjectModelMapGenerator(object):

    def __init__(self,
                 dataset_folder_path=None,
                 scannet_dataset_folder_path=None,
                 shapenet_dataset_folder_path=None,
                 scannet_object_dataset_folder_path=None,
                 scannet_bbox_dataset_folder_path=None):
        self.dataset_loader = DatasetLoader(dataset_folder_path,
                                            scannet_dataset_folder_path,
                                            shapenet_dataset_folder_path)

        self.scannet_object_dataset_folder_path = None
        self.scannet_bbox_dataset_folder_path = None
        self.shapenet_dataset_folder_path = None

        if None not in [
                dataset_folder_path, scannet_dataset_folder_path,
                shapenet_dataset_folder_path,
                scannet_object_dataset_folder_path,
                scannet_bbox_dataset_folder_path
        ]:
            self.loadDatasetFolderPath(dataset_folder_path,
                                       scannet_dataset_folder_path,
                                       shapenet_dataset_folder_path,
                                       scannet_object_dataset_folder_path,
                                       scannet_bbox_dataset_folder_path)
        return

    def reset(self):
        self.dataset_loader.reset()

        self.scannet_bbox_dataset_folder_path = None
        self.shapenet_dataset_folder_path = None
        return True

    def loadDatasetFolderPath(self, dataset_folder_path,
                              scannet_dataset_folder_path,
                              shapenet_dataset_folder_path,
                              scannet_object_dataset_folder_path,
                              scannet_bbox_dataset_folder_path):
        assert os.path.exists(dataset_folder_path)
        assert os.path.exists(scannet_dataset_folder_path)
        assert os.path.exists(shapenet_dataset_folder_path)
        assert os.path.exists(scannet_object_dataset_folder_path)
        assert os.path.exists(scannet_bbox_dataset_folder_path)

        self.reset()

        self.dataset_loader.loadDataset(dataset_folder_path,
                                        scannet_dataset_folder_path,
                                        shapenet_dataset_folder_path)

        self.scannet_object_dataset_folder_path = scannet_object_dataset_folder_path
        self.scannet_bbox_dataset_folder_path = scannet_bbox_dataset_folder_path
        self.shapenet_dataset_folder_path = shapenet_dataset_folder_path
        return True

    def generateSceneObjectModelMap(self, scene, save_map_json_file_path):
        scannet_scene_bbox_json_file_path = self.scannet_bbox_dataset_folder_path + \
            scene.scene_name + "/object_bbox.json"
        assert os.path.exists(scannet_scene_bbox_json_file_path)

        with open(scannet_scene_bbox_json_file_path, "r") as f:
            scannet_scene_bbox_json = json.load(f)

        scannet_object_position_dict = {}
        for scannet_object_file_name, scannet_object_bbox in scannet_scene_bbox_json.items(
        ):
            scannet_object_center = np.mean(scannet_object_bbox, axis=0)

            scannet_object_position_dict[scannet_object_file_name] = {
                'center': scannet_object_center,
                'bbox': scannet_object_bbox,
            }

        scene_object_model_map_dict = {}

        if False:
            # tmp saving source cad model files
            tmp_model_save_folder_path = "./tmp/source_shapenet_model/" + scene.scene_name + "/"
            for i, shapenet_model in enumerate(scene.model_list):
                shapenet_model_file_path = self.shapenet_dataset_folder_path + \
                    shapenet_model.cad_cat_id + "/" + shapenet_model.cad_id + "/models/model_normalized.obj"
                assert os.path.exists(shapenet_model_file_path)
                mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
                tmp_save_path = tmp_model_save_folder_path + str(i) + "_" + \
                    shapenet_model.cad_cat_id + "_" + shapenet_model.cad_id + ".ply"
                createFileFolder(tmp_save_path)
                o3d.io.write_triangle_mesh(tmp_save_path, mesh)

        if False:
            # tmp saving cad model files
            tmp_model_save_folder_path = "./tmp/shapenet_model/" + scene.scene_name + "/"
            for i, shapenet_model in enumerate(scene.model_list):
                shapenet_model_file_path = self.shapenet_dataset_folder_path + \
                    shapenet_model.cad_cat_id + "/" + shapenet_model.cad_id + "/models/model_normalized.obj"
                assert os.path.exists(shapenet_model_file_path)
                mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
                mesh.transform(shapenet_model.trans_model_to_scan_matrix)
                tmp_save_path = tmp_model_save_folder_path + str(i) + "_" + \
                    shapenet_model.cad_cat_id + "_" + shapenet_model.cad_id + ".ply"
                createFileFolder(tmp_save_path)
                o3d.io.write_triangle_mesh(tmp_save_path, mesh)

        if False:
            shapenet_model_file_path_list = []
            shapenet_model_trans_center_list = []
            for scannet_object_file_name, scannet_object_position_dict in scannet_object_position_dict.items(
            ):
                min_dist_shapenet_model_idx = -1
                min_dist_to_shapenet_model_trans_center = float('inf')

                for i in range(len(shapenet_model_file_path_list)):
                    shapenet_model_file_path = shapenet_model_file_path_list[i]
                    shapenet_model_trans_center = shapenet_model_trans_center_list[
                        i]

                    diff = scannet_object_center - shapenet_model_trans_center
                    current_dist_to_shapenet_model_trans_center = np.linalg.norm(
                        diff)
                    if current_dist_to_shapenet_model_trans_center < min_dist_to_shapenet_model_trans_center:
                        min_dist_shapenet_model_idx = i
                        min_dist_to_shapenet_model_trans_center = current_dist_to_shapenet_model_trans_center

                print(min_dist_shapenet_model_idx,
                      min_dist_to_shapenet_model_trans_center)
                scene_object_model_map_dict[
                    scannet_object_file_name] = scene.model_list[
                        min_dist_shapenet_model_idx].toDict()

        for shapenet_model in scene.model_list:
            shapenet_model_file_path = self.shapenet_dataset_folder_path + \
                shapenet_model.cad_cat_id + "/" + shapenet_model.cad_id + "/models/model_normalized.obj"
            assert os.path.exists(shapenet_model_file_path)

            shapenet_model_trans_center = np.mean(
                shapenet_model.trans_bbox_array, axis=0)
            shapenet_model_bbox = BBox()
            for bbox_point in shapenet_model.trans_bbox_array:
                point = Point.fromList(bbox_point)
                shapenet_model_bbox.addPoint(point)

            min_dist = float('inf')
            for scannet_object_file_name, scannet_object_position in scannet_object_position_dict.items(
            ):
                scannet_object_center = scannet_object_position['center']
                scannet_object_bbox = scannet_object_position['bbox']
                center_diff = np.linalg.norm(scannet_object_center -
                                             shapenet_model_trans_center)
                bbox_min_point_diff = np.linalg.norm(
                    scannet_object_bbox[0] -
                    shapenet_model_bbox.min_point.toArray())
                bbox_max_point_diff = np.linalg.norm(
                    scannet_object_bbox[1] -
                    shapenet_model_bbox.max_point.toArray())
                current_dist = center_diff + bbox_min_point_diff + bbox_max_point_diff

                print("dist to ", scannet_object_file_name, "is", current_dist)

                if current_dist < min_dist:
                    min_dist_scannet_object_file_name = scannet_object_file_name
                    min_dist = current_dist

            print(min_dist_scannet_object_file_name)
            scene_object_model_map_dict[
                min_dist_scannet_object_file_name] = shapenet_model.toDict()

        createFileFolder(save_map_json_file_path)
        with open(save_map_json_file_path, "w") as f:
            json.dump(scene_object_model_map_dict, f, indent=4)
        return True

    def generateAllSceneObjectModelMap(self,
                                       save_map_json_folder_path,
                                       print_progress=False):
        os.makedirs(save_map_json_folder_path, exist_ok=True)

        for_data = self.dataset_loader.dataset.scene_dict.items()
        if print_progress:
            print(
                "[INFO][ObjectModelMapGenerator::generateAllSceneObjectModelMap]"
            )
            print("\t start generate object model map json for all scenes...")
            for_data = tqdm(for_data)
        for scene_name, scene in for_data:
            if scene_name != "scene0474_02":
                continue
            save_map_json_file_path = save_map_json_folder_path + \
                scene_name + "/object_model_map.json"

            removeFile(save_map_json_file_path)

            if os.path.exists(save_map_json_file_path):
                continue

            tmp_save_map_json_file_path = save_map_json_file_path[:-5] + "_tmp.json"
            self.generateSceneObjectModelMap(scene,
                                             tmp_save_map_json_file_path)

            renameFile(tmp_save_map_json_file_path, save_map_json_file_path)
            return True
        return True
