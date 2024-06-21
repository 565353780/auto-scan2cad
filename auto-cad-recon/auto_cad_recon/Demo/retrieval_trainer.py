#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage/")
sys.path.append("../udf-generate/")
sys.path.append("../scannet-dataset-manage")
sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../shapenet-dataset-manage")
sys.path.append("../points-shape-detect")
sys.path.append("../global-pose-refine")
sys.path.append("../conv-onet")

from auto_cad_recon.Module.retrieval_trainer import RetrievalTrainer


def demo():
    scannet_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
    scannet_object_dataset_folder_path = "/home/chli/chLi/ScanNet/objects/"
    scannet_bbox_dataset_folder_path = "/home/chli/chLi/ScanNet/bboxes/"
    scan2cad_dataset_folder_path = "/home/chli/chLi/Scan2CAD/scan2cad_dataset/"
    scan2cad_object_model_map_dataset_folder_path = "/home/chli/chLi/Scan2CAD/object_model_maps/"
    shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
    shapenet_udf_dataset_folder_path = "/home/chli/chLi/ShapeNet/udfs/"
    #  model_file_path = "./output/20221026_14:41:17/model_last.pth"
    model_file_path = "./output/20221026_19:41:08/model_last.pth"
    print_progress = True

    retrieval_trainer = RetrievalTrainer(
        scannet_dataset_folder_path, scannet_object_dataset_folder_path,
        scannet_bbox_dataset_folder_path, scan2cad_dataset_folder_path,
        scan2cad_object_model_map_dataset_folder_path,
        shapenet_dataset_folder_path, shapenet_udf_dataset_folder_path)

    retrieval_trainer.loadModel(model_file_path)

    retrieval_trainer.train(print_progress)
    return True
