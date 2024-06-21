#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage")
sys.path.append("../habitat-sim-manage")
sys.path.append("../scannet-sim-manage")
sys.path.append("../detectron2-detect")
sys.path.append("../scene-layout-detect")

from auto_cad_recon.Module.dataset_render_manager import DatasetRenderManager


def demo():
    scannet_glb_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
    scannet_object_dataset_folder_path = "/home/chli/chLi/ScanNet/objects/"
    scannet_bbox_dataset_folder_path = "/home/chli/chLi/ScanNet/bboxes/"
    control_mode = "pose"
    wait_val = 1
    print_progress = True

    scannet_scene_name = "scene0474_02"
    scene_objects_save_folder_path = "./output/scene_objects/scene0474_02/"

    dataset_render_manager = DatasetRenderManager(
        scannet_glb_dataset_folder_path, scannet_object_dataset_folder_path,
        scannet_bbox_dataset_folder_path)
    dataset_render_manager.loadScene(scannet_scene_name, print_progress)
    dataset_render_manager.setControlMode(control_mode)

    dataset_render_manager.setAgentPose([3.7, 1.0, -2.6], [0.2, 0.0, 0.0])

    dataset_render_manager.startKeyBoardControlRender(wait_val, print_progress)

    scene_object_label_list = dataset_render_manager.getSceneObjectLabelList()
    print("scene_object_label_list :")
    print(scene_object_label_list)

    if len(scene_object_label_list) > 0:
        frame_object_dict = dataset_render_manager.getFrameObjectDict(
            scene_object_label_list[0])
        print("frame_object_dict.keys :")
        print(frame_object_dict.keys())

    dataset_render_manager.saveAllSceneObjects(scene_objects_save_folder_path)
    return True
