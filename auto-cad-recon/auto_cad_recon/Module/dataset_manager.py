#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np

from scan2cad_dataset_manage.Module.object_model_map_generator import ObjectModelMapGenerator
from scan2cad_dataset_manage.Module.object_model_map_manager import ObjectModelMapManager

from scannet_dataset_manage.Module.object_bbox_generator import ObjectBBoxGenerator
from scannet_dataset_manage.Module.object_spliter import ObjectSpliter

from shapenet_dataset_manage.Module.image_format_fixer import ImageFormatFixer

from udf_generate.Module.udf_generate_manager import UDFGenerateManager


class DatasetManager(object):

    def __init__(self, scannet_dataset_folder_path,
                 scannet_object_dataset_folder_path,
                 scannet_bbox_dataset_folder_path,
                 scan2cad_dataset_folder_path,
                 scan2cad_object_model_map_dataset_folder_path,
                 shapenet_dataset_folder_path,
                 shapenet_udf_dataset_folder_path):
        self.scannet_dataset_folder_path = scannet_dataset_folder_path
        self.scannet_object_dataset_folder_path = scannet_object_dataset_folder_path
        self.scannet_bbox_dataset_folder_path = scannet_bbox_dataset_folder_path
        self.scan2cad_dataset_folder_path = scan2cad_dataset_folder_path
        self.scan2cad_object_model_map_dataset_folder_path = scan2cad_object_model_map_dataset_folder_path
        self.shapenet_dataset_folder_path = shapenet_dataset_folder_path
        self.shapenet_udf_dataset_folder_path = shapenet_udf_dataset_folder_path

        self.object_model_map_manager = ObjectModelMapManager(
            self.scannet_object_dataset_folder_path,
            self.shapenet_dataset_folder_path,
            self.scan2cad_object_model_map_dataset_folder_path)
        return

    def splitScanNetObject(self):
        object_spliter = ObjectSpliter(self.scannet_dataset_folder_path,
                                       self.scannet_object_dataset_folder_path)
        object_spliter.splitAll()
        return True

    def generateScanNetObjectBBox(self):
        object_bbox_generator = ObjectBBoxGenerator(
            self.scannet_object_dataset_folder_path,
            self.scannet_bbox_dataset_folder_path)
        object_bbox_generator.generateObjectBBoxJson()
        return True

    def generateScan2CADObjectModelMap(self, print_progress=False):
        object_model_map_generator = ObjectModelMapGenerator(
            self.scan2cad_dataset_folder_path,
            self.scannet_dataset_folder_path,
            self.shapenet_dataset_folder_path,
            self.scannet_object_dataset_folder_path,
            self.scannet_bbox_dataset_folder_path)
        object_model_map_generator.generateAllSceneObjectModelMap(
            self.scan2cad_object_model_map_dataset_folder_path, print_progress)
        return True

    def fixShapeNetImageFormat(self, print_progress=False):
        image_format_fixer = ImageFormatFixer(
            self.shapenet_dataset_folder_path)
        image_format_fixer.fixAllImageFormat(print_progress)
        return True

    def generateShapeNetUDF(self):
        udf_generate_manager = UDFGenerateManager(
            self.shapenet_dataset_folder_path)
        udf_generate_manager.activeGenerateAllUDF(
            self.shapenet_udf_dataset_folder_path)
        return True

    def generateFullDataset(self, print_progress=False):
        self.splitScanNetObject()
        self.generateScanNetObjectBBox()
        self.generateScan2CADObjectModelMap(print_progress)
        self.fixShapeNetImageFormat(print_progress)
        self.generateShapeNetUDF()
        return True

    def getScanNetSceneNameList(self):
        return self.object_model_map_manager.scene_name_list

    def getScanNetObjectFileNameList(self, scannet_scene_name):
        return self.object_model_map_manager.getObjectFileNameList(
            scannet_scene_name)

    def isSceneValid(self, scannet_scene_name):
        return self.object_model_map_manager.isSceneValid(scannet_scene_name)

    def isObjectValid(self, scannet_scene_name, scannet_object_file_name):
        return self.object_model_map_manager.isObjectValid(
            scannet_scene_name, scannet_object_file_name)

    def getShapeNetModelDict(self, scannet_scene_name,
                             scannet_object_file_name):
        if not self.object_model_map_manager.isSceneValid(scannet_scene_name):
            return None

        if not self.object_model_map_manager.isObjectValid(
                scannet_scene_name, scannet_object_file_name):
            return None

        shapenet_model_dict = self.object_model_map_manager.getShapeNetModelDict(
            scannet_scene_name, scannet_object_file_name)

        shapenet_udf_folder_path = self.shapenet_udf_dataset_folder_path + shapenet_model_dict[
            'cad_cat_id'] + "/" + shapenet_model_dict[
                'cad_id'] + "/models/model_normalized/"
        assert os.path.exists(shapenet_udf_folder_path)

        shapenet_model_dict[
            'shapenet_udf_folder_path'] = shapenet_udf_folder_path

        shapenet_model_dict['cad_udf'] = np.load(shapenet_udf_folder_path +
                                                 "udf.npy").tolist()
        return shapenet_model_dict

    def getShapeNetModelTensorDict(self, scannet_scene_name,
                                   scannet_object_file_name,
                                   reshape=True):
        shapenet_model_dict = self.getShapeNetModelDict(
            scannet_scene_name, scannet_object_file_name)
        shapenet_model_tensor_dict = {}
        for key, item in shapenet_model_dict.items():
            if isinstance(item, list):
                tensor_item = torch.tensor(item).type(torch.FloatTensor)
                if reshape:
                    tensor_item = tensor_item.reshape(-1, *tensor_item.shape)
                shapenet_model_tensor_dict[key] = tensor_item
            else:
                shapenet_model_tensor_dict[key] = item
        return shapenet_model_tensor_dict
