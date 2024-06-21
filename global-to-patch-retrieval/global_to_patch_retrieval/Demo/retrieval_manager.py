#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../conv-onet")
sys.path.append("../points-shape-detect")
sys.path.append("../noc-transform")
sys.path.append("../scene-layout-detect")

from global_to_patch_retrieval.Module.retrieval_manager import RetrievalManager


def demo():
    shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
    shapenet_feature_folder_path = "/home/chli/chLi/ShapeNet/features/"
    obb_info_folder_path = "/home/chli/chLi/auto-scan2cad/1314/obb_info/"
    render = True
    print_progress = True

    retrieval_manager = RetrievalManager(shapenet_dataset_folder_path,
                                         print_progress)

    #  retrieval_manager.generateAllCADFeature(shapenet_feature_folder_path,
    #  print_progress)

    retrieval_manager.generateRetrievalResult(obb_info_folder_path,
                                              shapenet_feature_folder_path,
                                              render, print_progress)
    return True
