#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np

sys.path.append("../mesh-manage")

from scan2cad_dataset_manage.Module.dataset_loader import DatasetLoader


def demo():
    dataset_folder_path = "/home/chli/chLi/Scan2CAD/scan2cad_dataset/"
    scannet_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
    shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
    save_cad_folder_path = "./output/cad/"
    print_progress = True

    render_scene_by_obj_num = False
    save_all_valid_scene_cad_models = False

    dataset_loader = DatasetLoader(dataset_folder_path,
                                   scannet_dataset_folder_path,
                                   shapenet_dataset_folder_path)

    scene_name_list = dataset_loader.getSceneNameList()

    model_num_list = []

    for scene_name in scene_name_list:
        model_num_list.append(
            len(dataset_loader.dataset.scene_dict[scene_name].model_list))

    valid_scene_name_list = [
        'scene0474_02', 'scene0000_01', 'scene0667_01', 'scene0500_00',
        'scene0247_01', 'scene0644_00', 'scene0231_01', 'scene0653_00',
        'scene0300_00', 'scene0569_00', 'scene0588_01', 'scene0603_01',
        'scene0054_00', 'scene0673_04'
    ]
    valid_scene_list = [
        scene_name.split("_")[0] for scene_name in valid_scene_name_list
    ]

    # rank by object num and render
    if render_scene_by_obj_num:
        for _ in range(len(scene_name_list)):
            scene_idx = np.argmax(model_num_list)
            scene_name = scene_name_list[scene_idx]
            model_num = len(
                dataset_loader.dataset.scene_dict[scene_name].model_list)
            assert model_num == model_num_list[scene_idx]
            model_num_list[scene_idx] = -1

            if scene_name.split("_")[0] in valid_scene_list:
                continue

            dataset_loader.renderScan2CADScene(scene_name)
            print(scene_name, "->", model_num, "objects")
        return True

    # saving valid scene cad models
    if save_all_valid_scene_cad_models:
        valid_scene_name_list = valid_scene_name_list[-4:]

        for scene_name in valid_scene_name_list:
            #  dataset_loader.renderScan2CADScene(scene_name)
            dataset_loader.saveSceneCAD(scene_name, save_cad_folder_path,
                                        print_progress)
        return True

    #  scannet_scene_name = "scene0013_02"
    #  scannet_scene_name = "scene0474_02"
    scannet_scene_name = valid_scene_name_list[10]
    assert dataset_loader.isSceneValid(scannet_scene_name)

    dataset_loader.renderScan2CADScene(scannet_scene_name)

    dataset_loader.saveSceneCAD(scannet_scene_name, save_cad_folder_path,
                                print_progress)
    return True
