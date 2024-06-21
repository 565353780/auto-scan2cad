#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage/")
sys.path.append("../udf-generate/")
sys.path.append("../scannet-dataset-manage")
sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../shapenet-dataset-manage")

from auto_cad_recon.Module.retrieval_manager import RetrievalManager

#  conv_onet_param = "0.01_55_8crop_encode"
#  conv_onet_param = "0.01_44_27crop_encode"
#  conv_onet_param = "0.01_44_27crop_occ"
#  conv_onet_param = "0.01_55_8crop_occ"
#  conv_onet_param = "0.01_44_27crop_occ"
conv_onet_param = "0.1_1_1000crop_occ_oocc_weight"
#  conv_onet_param = "0.1_1_1000crop_occ_oocc_weight_noise"
#  conv_onet_param = "32x32x32_udf"
#  conv_onet_param = "10x10x10_udf"

mode_list = ['conv_onet_encode', 'conv_onet_occ', 'occ', 'occ_noise', 'udf']
mode = 'occ'


def demo():
    scannet_scene_name = "scene0474_02"
    save_pkl_file_path = "/home/chli/chLi/auto_cad_recon/error_matrix/" + \
        conv_onet_param + "/scan_to_cad_error_matrix.npy"

    retrieval_manager = RetrievalManager(mode)
    retrieval_manager.loadScene(scannet_scene_name)
    retrieval_manager.generateErrorMatrix(save_pkl_file_path)
    return True


def demo_render():
    scannet_scene_name = "scene0474_02"
    scan_to_cad_error_matrix_pkl_file_path = "/home/chli/chLi/auto_cad_recon/error_matrix/" + \
        conv_onet_param + "/scan_to_cad_error_matrix.npy"

    retrieval_manager = RetrievalManager(mode)
    retrieval_manager.loadScene(scannet_scene_name, False)
    retrieval_manager.renderRetrievalResult(
        scan_to_cad_error_matrix_pkl_file_path)
    return True
