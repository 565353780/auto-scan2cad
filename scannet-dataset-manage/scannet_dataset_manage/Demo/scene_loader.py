#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scannet_dataset_manage.Module.scene_loader import SceneLoader


def demo():
    scene_folder_path = "/home/chli/chLi/ScanNet/scans/scene0000_00/"

    scene_loader = SceneLoader(scene_folder_path)
    labeled_object_num = scene_loader.getLabeledObjectNum()
    for i in range(labeled_object_num):
        point_idx_list = scene_loader.getPointIdxListByLabeledObjectId(i)
        print("object", i, "have", len(point_idx_list), "points")
    return True
