#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from shutil import copyfile

import open3d as o3d
from mesh_manage.Module.channel_mesh import ChannelMesh
from tqdm import tqdm

from scan2cad_dataset_manage.Data.dataset import Dataset
from scan2cad_dataset_manage.Method.bbox import getNearestModelIdxByBBoxDist
from scan2cad_dataset_manage.Method.render import renderScan2CADScene


class DatasetLoader(object):

    def __init__(self,
                 dataset_folder_path=None,
                 scannet_dataset_folder_path=None,
                 shapenet_dataset_folder_path=None):
        self.dataset_folder_path = None
        self.scannet_dataset_folder_path = None
        self.shapenet_dataset_folder_path = None

        self.cad_appearances_json = None
        self.unique_cad_csv = None

        self.dataset = None

        if None not in [
                dataset_folder_path, scannet_dataset_folder_path,
                shapenet_dataset_folder_path
        ]:
            self.loadDataset(dataset_folder_path, scannet_dataset_folder_path,
                             shapenet_dataset_folder_path)
        return

    def reset(self):
        self.dataset_folder_path = None
        self.scannet_dataset_folder_path = None
        self.shapenet_dataset_folder_path = None

        self.cad_appearances_json = None
        self.unique_cad_csv = None

        if self.dataset is not None:
            del self.dataset
            self.dataset = None
        return True

    def loadDataset(self, dataset_folder_path, scannet_dataset_folder_path,
                    shapenet_dataset_folder_path):
        assert os.path.exists(dataset_folder_path)
        assert os.path.exists(scannet_dataset_folder_path)
        assert os.path.exists(shapenet_dataset_folder_path)

        self.reset()

        self.dataset_folder_path = dataset_folder_path
        self.scannet_dataset_folder_path = scannet_dataset_folder_path
        self.shapenet_dataset_folder_path = shapenet_dataset_folder_path

        #  cad_appearances_json_file_path = self.dataset_folder_path + "cad_appearances.json"
        #  assert os.path.exists(cad_appearances_json_file_path)
        #  with open(cad_appearances_json_file_path, "r") as f:
        #     self.cad_appearances_json = json.load(f)

        full_annotations_json_file_path = self.dataset_folder_path + "full_annotations.json"
        assert os.path.exists(full_annotations_json_file_path)

        with open(full_annotations_json_file_path, "r") as f:
            scene_dict_list = json.load(f)

        self.dataset = Dataset(scene_dict_list)
        return True

    def getSceneNameList(self):
        return list(self.dataset.scene_dict.keys())

    def isSceneValid(self, scene_name):
        return scene_name in self.dataset.scene_dict.keys()

    def getScanNetSceneFolderPath(self, scannet_scene_name):
        assert scannet_scene_name in self.dataset.scene_dict.keys()
        return self.scannet_dataset_folder_path + scannet_scene_name

    def getShapeNetModelFilePath(self, scannet_scene_name, model_idx):
        assert scannet_scene_name in self.dataset.scene_dict.keys()
        scene = self.dataset.scene_dict[scannet_scene_name]

        assert model_idx < len(scene.model_list)
        model = scene.model_list[model_idx]
        shapenet_model_file_path = self.shapenet_dataset_folder_path + \
            model.cad_cat_id + "/" + model.cad_id + "/models/model_normalized.obj"
        return shapenet_model_file_path

    def getShapeNetModelFilePathByBBoxDist(self, scannet_scene_name,
                                           scannet_object_file_path):
        assert scannet_scene_name in self.dataset.scene_dict.keys()

        object_bbox = ChannelMesh(scannet_object_file_path).getBBox()

        min_bbox_dist_model_idx = getNearestModelIdxByBBoxDist(
            object_bbox, self.dataset.scene_dict[scannet_scene_name])

        shapenet_model_file_path = self.getShapeNetModelFilePath(
            scannet_scene_name, min_bbox_dist_model_idx)
        assert os.path.exists(shapenet_model_file_path)
        return shapenet_model_file_path

    def renderScan2CADScene(self, scannet_scene_name):
        assert scannet_scene_name in self.dataset.scene_dict.keys()

        scannet_scene_file_path = self.scannet_dataset_folder_path + \
            scannet_scene_name + "/" + scannet_scene_name + "_vh_clean.ply"
        assert os.path.exists(scannet_scene_file_path)

        renderScan2CADScene(self.dataset.scene_dict[scannet_scene_name],
                            scannet_scene_file_path)
        return True

    def saveSceneCAD(self,
                     scannet_scene_name,
                     save_folder_path,
                     print_progress=False):
        assert scannet_scene_name in self.dataset.scene_dict.keys()

        scene = self.dataset.scene_dict[scannet_scene_name]

        save_folder_path = save_folder_path + scannet_scene_name + "/"

        os.makedirs(save_folder_path, exist_ok=True)

        scene_mesh_file_path = self.scannet_dataset_folder_path + scannet_scene_name + "/" + scannet_scene_name + "_vh_clean_2.ply"
        assert os.path.exists(scene_mesh_file_path)

        copyfile(scene_mesh_file_path, save_folder_path + "scene_mesh.ply")

        for_data = enumerate(scene.model_list)
        if print_progress:
            print("[INFO][DatasetLoader::saveSceneCAD]")
            print("\t start save scene cad models...")
            for_data = tqdm(for_data, total=len(scene.model_list))
        for i, shapenet_model in for_data:
            shapenet_model_file_path = self.shapenet_dataset_folder_path + \
                shapenet_model.cad_cat_id + "/" + shapenet_model.cad_id + \
                "/models/model_normalized.obj"
            assert os.path.exists(shapenet_model_file_path)
            shapenet_model_mesh = o3d.io.read_triangle_mesh(
                shapenet_model_file_path)
            shapenet_model_mesh.transform(
                shapenet_model.trans_model_to_scan_matrix)

            shapenet_model_mesh.compute_vertex_normals()

            save_file_path = save_folder_path + str(i) + ".ply"

            o3d.io.write_triangle_mesh(save_file_path,
                                       shapenet_model_mesh,
                                       write_ascii=True)
        return True
