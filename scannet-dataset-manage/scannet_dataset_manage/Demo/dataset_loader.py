#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scannet_dataset_manage.Module.dataset_loader import DatasetLoader


def demo():
    dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"

    dataset_loader = DatasetLoader(dataset_folder_path)
    scene_num = dataset_loader.getSceneNum()
    print("scene_num =", scene_num)
    scene_name_list = dataset_loader.getSceneNameList()
    print("scene_name_list =", scene_name_list)
    return True
