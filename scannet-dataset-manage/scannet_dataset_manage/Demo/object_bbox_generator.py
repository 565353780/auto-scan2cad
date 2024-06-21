#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage/")

from scannet_dataset_manage.Module.object_bbox_generator import ObjectBBoxGenerator


def demo():
    objects_folder_path = "/home/chli/chLi/ScanNet/objects/"
    save_json_folder_path = "/home/chli/chLi/ScanNet/bboxes/"

    object_bbox_generator = ObjectBBoxGenerator(objects_folder_path,
                                                save_json_folder_path)
    object_bbox_generator.generateObjectBBoxJson()
    return True
