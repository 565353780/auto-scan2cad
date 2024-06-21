#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage/")
sys.path.append("../udf-generate/")
sys.path.append("../scannet-dataset-manage")
sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../shapenet-dataset-manage")

from auto_cad_recon.Module.dataset_manager import DatasetManager


def demo():
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

    #  dataset_manager.generateFullDataset(print_progress)

    scene_name_list = dataset_manager.getScanNetSceneNameList()
    print("scene_name_list num =", len(scene_name_list))

    scannet_scene_name = "scene0474_02"
    assert dataset_manager.isSceneValid(scannet_scene_name)

    object_file_name_list = dataset_manager.getScanNetObjectFileNameList(
        scannet_scene_name)
    print("object_file_name_list for scene " + scannet_scene_name + ":")
    print(object_file_name_list)

    scannet_object_file_name = object_file_name_list[0]
    assert dataset_manager.isObjectValid(scannet_scene_name,
                                         scannet_object_file_name)

    shapenet_model_dict = dataset_manager.getShapeNetModelDict(
        scannet_scene_name, scannet_object_file_name)

    print(shapenet_model_dict)

    return True
