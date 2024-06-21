#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage/")

from scannet_dataset_manage.Module.object_spliter import ObjectSpliter


def demo():
    dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
    save_object_folder_path = "/home/chli/chLi/ScanNet/objects/"

    object_spliter = ObjectSpliter(dataset_folder_path,
                                   save_object_folder_path)
    object_spliter.splitAll()
    return True
