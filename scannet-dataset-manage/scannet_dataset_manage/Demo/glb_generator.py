#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scannet_dataset_manage.Module.glb_generator import GLBGenerator


def demo():
    scannet_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
    scannet_glb_dataset_folder_path = "/home/chli/chLi/ScanNet/glb/"
    print_progress = True

    glb_generator = GLBGenerator(scannet_dataset_folder_path,
                                 scannet_glb_dataset_folder_path)
    glb_generator.generateAll(print_progress)
    return True
