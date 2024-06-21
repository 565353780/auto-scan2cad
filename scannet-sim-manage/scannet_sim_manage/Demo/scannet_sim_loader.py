#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage")
sys.path.append("../habitat-sim-manage")
sys.path.append("../detectron2-detect")
sys.path.append("../scene-layout-detect")

from scannet_sim_manage.Module.scannet_sim_loader import ScanNetSimLoader


def demo():
    glb_file_path = \
        "/home/chli/chLi/ScanNet/scans/scene0474_02/scene0474_02_vh_clean.glb"
    object_folder_path = "/home/chli/chLi/ScanNet/objects/scene0474_02/"
    bbox_json_file_path = "/home/chli/chLi/ScanNet/bboxes/scene0474_02/object_bbox.json"
    control_mode = "pose"
    wait_val = 1
    bbox_image_width = 127
    bbox_image_height = 127
    bbox_free_width = 5
    print_progress = True

    scene_objects_save_folder_path = "./output/scene_objects/scene0474_02/"

    scannet_sim_loader = ScanNetSimLoader()
    scannet_sim_loader.loadScene(glb_file_path, object_folder_path,
                                 bbox_json_file_path, print_progress)
    scannet_sim_loader.setControlMode(control_mode)

    scannet_sim_loader.setAgentPose([3.7, 1.0, -2.6], [0.2, 0.0, 0.0])

    scannet_sim_loader.startKeyBoardControlRender(wait_val)

    scannet_sim_loader.saveAllSceneObjects(scene_objects_save_folder_path,
                                           bbox_image_width, bbox_image_height,
                                           bbox_free_width)
    return True
