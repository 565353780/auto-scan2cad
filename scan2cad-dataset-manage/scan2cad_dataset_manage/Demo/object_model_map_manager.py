#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scan2cad_dataset_manage.Module.object_model_map_manager import ObjectModelMapManager


def demo():
    scannet_object_dataset_folder_path = "/home/chli/chLi/ScanNet/objects/"
    shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
    object_model_map_dataset_folder_path = "/home/chli/chLi/Scan2CAD/object_model_maps/"

    object_model_map_manager = ObjectModelMapManager(
        scannet_object_dataset_folder_path, shapenet_dataset_folder_path,
        object_model_map_dataset_folder_path)

    scannet_scene_name = "scene0013_02"
    scannet_scene_name = "scene0474_02"
    assert object_model_map_manager.isSceneValid(scannet_scene_name)

    object_file_name_list = object_model_map_manager.getObjectFileNameList(scannet_scene_name)
    print("object_file_name_list:")
    print(object_file_name_list)

    scannet_object_file_name = object_file_name_list[0]
    assert object_model_map_manager.isObjectValid(scannet_scene_name, scannet_object_file_name)

    shapenet_model_dict = object_model_map_manager.getShapeNetModelDict(
        scannet_scene_name, scannet_object_file_name)

    print(shapenet_model_dict)
    print(shapenet_model_dict.keys())

    object_model_map_manager.renderScan2CADObjectModelMap(scannet_scene_name, scannet_object_file_name)
    return True
