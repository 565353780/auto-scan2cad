#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage")

from scan2cad_dataset_manage.Module.object_model_map_generator import ObjectModelMapGenerator


def demo():
    dataset_folder_path = "/home/chli/chLi/Scan2CAD/scan2cad_dataset/"
    scannet_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
    shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
    scannet_object_dataset_folder_path = "/home/chli/chLi/ScanNet/objects/"
    scannet_bbox_dataset_folder_path = "/home/chli/chLi/ScanNet/bboxes/"
    save_map_json_folder_path = "/home/chli/chLi/Scan2CAD/object_model_maps/"
    print_progress = True

    object_model_map_generator = ObjectModelMapGenerator(
        dataset_folder_path, scannet_dataset_folder_path,
        shapenet_dataset_folder_path, scannet_object_dataset_folder_path,
        scannet_bbox_dataset_folder_path)

    object_model_map_generator.generateAllSceneObjectModelMap(
        save_map_json_folder_path, print_progress)
    return True
