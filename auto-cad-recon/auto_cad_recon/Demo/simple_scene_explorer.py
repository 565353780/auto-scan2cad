#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage/")
sys.path.append("../udf-generate/")
sys.path.append("../scannet-dataset-manage")
sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../shapenet-dataset-manage")
sys.path.append("../pytorch-3d-r2n2")
sys.path.append("../conv-onet")
sys.path.append("../detectron2-detect")
sys.path.append("../scene-layout-detect")
sys.path.append("../points-shape-detect")
sys.path.append("../global-pose-refine")
sys.path.append("../noc-transform")

from auto_cad_recon.Module.simple_scene_explorer import SimpleSceneExplorer


def demo():
    scannet_dataset_folder_path = "/home/chli/chLi/Dataset/ScanNet/scans/"
    scannet_glb_dataset_folder_path = "/home/chli/chLi/Dataset/ScanNet/glb/"
    scannet_object_dataset_folder_path = "/home/chli/chLi/Dataset/ScanNet/objects/"
    scannet_bbox_dataset_folder_path = "/home/chli/chLi/Dataset/ScanNet/bboxes/"
    scan2cad_dataset_folder_path = "/home/chli/chLi/Dataset/Scan2CAD/scan2cad_dataset/"
    scan2cad_object_model_map_dataset_folder_path = (
        "/home/chli/chLi/Dataset/Scan2CAD/object_model_maps/"
    )
    shapenet_dataset_folder_path = (
        "/home/chli/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/"
    )
    shapenet_udf_dataset_folder_path = "/home/chli/chLi/Dataset/ShapeNet/udfs/"
    control_mode = "pose"
    wait_val = 1
    bbox_image_width = 127
    bbox_image_height = 127
    bbox_image_free_width = 5
    print_progress = True

    simple_scene_explorer = SimpleSceneExplorer(
        scannet_dataset_folder_path,
        scannet_glb_dataset_folder_path,
        scannet_object_dataset_folder_path,
        scannet_bbox_dataset_folder_path,
        scan2cad_dataset_folder_path,
        scan2cad_object_model_map_dataset_folder_path,
        shapenet_dataset_folder_path,
        shapenet_udf_dataset_folder_path,
    )

    #  simple_scene_explorer.generateFullDataset(print_progress)

    #  simple_scene_explorer.loadRetrievalNetModel(retrieval_net_model_file_path)

    scene_name_list = simple_scene_explorer.getScanNetSceneNameList()
    print("scene_name_list num =", len(scene_name_list))

    valid_scene_name_list = [
        "scene0474_02",
        "scene0000_01",
        "scene0667_01",
        "scene0500_00",
        "scene0247_01",
        "scene0644_00",
        "scene0231_01",
        "scene0653_00",
        "scene0300_00",
        "scene0569_00",
        "scene0588_01",
        "scene0603_01",
        "scene0054_00",
        "scene0673_04",
    ]

    scannet_scene_name = valid_scene_name_list[7]
    assert simple_scene_explorer.isSceneValid(scannet_scene_name)

    object_file_name_list = simple_scene_explorer.getScanNetObjectFileNameList(
        scannet_scene_name
    )
    print("object_file_name_list for scene " + scannet_scene_name + ":")

    scannet_object_file_name = object_file_name_list[0]
    assert simple_scene_explorer.isObjectValid(
        scannet_scene_name, scannet_object_file_name
    )

    shapenet_model_dict = simple_scene_explorer.getShapeNetModelDict(
        scannet_scene_name, scannet_object_file_name
    )
    print("shapenet_model_dict :")
    print(shapenet_model_dict.keys())

    simple_scene_explorer.loadScene(scannet_scene_name, print_progress)
    simple_scene_explorer.setControlMode(control_mode)

    simple_scene_explorer.setAgentPose([3.7, 1.0, -2.6], [0.2, 0.0, 0.0])

    simple_scene_explorer.startKeyBoardControlRender(wait_val, print_progress)

    scene_object_label_list = simple_scene_explorer.getSceneObjectLabelList()
    print("scene_object_label_list :")
    print(scene_object_label_list)

    if len(scene_object_label_list) > 0:
        frame_object_dict = simple_scene_explorer.getFrameObjectDict(
            scene_object_label_list[0]
        )
        print("frame_object_dict.keys :")
        print(frame_object_dict.keys())

    scene_objects_save_folder_path = (
        "./output/scene_objects/" + scannet_scene_name + "/"
    )
    simple_scene_explorer.saveAllSceneObjects(
        scene_objects_save_folder_path,
        bbox_image_width,
        bbox_image_height,
        bbox_image_free_width,
    )
    return True
