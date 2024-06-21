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

from auto_cad_recon.Module.auto_cad_reconstructor import AutoCADReconstructor


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
    retrieval_net_model_file_path = "/home/chli/chLi/auto-scan2cad/my_model/global_to_local_retrieval/output/pretrained_retrieval/model_last.pth"
    control_mode = "pose"
    wait_val = 1
    bbox_image_width = 127
    bbox_image_height = 127
    bbox_image_free_width = 5
    print_progress = True

    auto_cad_reconstructor = AutoCADReconstructor(
        scannet_dataset_folder_path,
        scannet_glb_dataset_folder_path,
        scannet_object_dataset_folder_path,
        scannet_bbox_dataset_folder_path,
        scan2cad_dataset_folder_path,
        scan2cad_object_model_map_dataset_folder_path,
        shapenet_dataset_folder_path,
        shapenet_udf_dataset_folder_path,
    )

    #  auto_cad_reconstructor.generateFullDataset(print_progress)

    #  auto_cad_reconstructor.loadRetrievalNetModel(retrieval_net_model_file_path)

    scene_name_list = auto_cad_reconstructor.getScanNetSceneNameList()
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

    scannet_scene_name = valid_scene_name_list[1]
    assert auto_cad_reconstructor.isSceneValid(scannet_scene_name)

    object_file_name_list = auto_cad_reconstructor.getScanNetObjectFileNameList(
        scannet_scene_name
    )
    print("object_file_name_list for scene " + scannet_scene_name + ":")

    scannet_object_file_name = object_file_name_list[0]
    assert auto_cad_reconstructor.isObjectValid(
        scannet_scene_name, scannet_object_file_name
    )

    shapenet_model_dict = auto_cad_reconstructor.getShapeNetModelDict(
        scannet_scene_name, scannet_object_file_name
    )
    print("shapenet_model_dict :")
    print(shapenet_model_dict.keys())

    auto_cad_reconstructor.loadScene(scannet_scene_name, print_progress)
    auto_cad_reconstructor.setControlMode(control_mode)

    auto_cad_reconstructor.setAgentPose([3.7, 1.0, -2.6], [0.2, 0.0, 0.0])

    auto_cad_reconstructor.startKeyBoardControlRender(wait_val, print_progress)

    scene_object_label_list = auto_cad_reconstructor.getSceneObjectLabelList()
    print("scene_object_label_list :")
    print(scene_object_label_list)

    if len(scene_object_label_list) > 0:
        frame_object_dict = auto_cad_reconstructor.getFrameObjectDict(
            scene_object_label_list[0]
        )
        print("frame_object_dict.keys :")
        print(frame_object_dict.keys())

    scene_objects_save_folder_path = (
        "./output/scene_objects/" + scannet_scene_name + "/"
    )
    auto_cad_reconstructor.saveAllSceneObjects(
        scene_objects_save_folder_path,
        bbox_image_width,
        bbox_image_height,
        bbox_image_free_width,
    )
    return True
