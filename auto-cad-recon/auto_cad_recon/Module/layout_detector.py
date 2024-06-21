#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2
import numpy as np
import open3d as o3d
import torch
from net_utils.libs import get_layout_bdb_sunrgbd, get_rotation_matix_result
from tqdm import tqdm
from utils.visualize import Box, format_layout

from auto_cad_recon.Model.layout.layout_net import LayoutNet
from configs.data_config import Config


class LayoutDetector(object):

    def __init__(self, model_file_path=None):
        self.model = LayoutNet().cuda()

        config = Config('sunrgbd')
        self.bins_tensor = {}
        for k, v in config.bins.items():
            if isinstance(v, list):
                self.bins_tensor[k] = torch.tensor(v).cuda()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        assert os.path.exists(model_file_path)

        print("[INFO][LayoutDetector::loadModel]")
        print("\t start loading model from :")
        print("\t", model_file_path)
        model_dict = torch.load(model_file_path)
        self.model.load_state_dict(model_dict['model'])
        return True

    def detectImage(self, image):
        self.model.eval()

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        resized_image = cv2.resize(image, (256, 256))
        resized_image = resized_image[..., ::-1].copy()
        query_image = torch.from_numpy(resized_image.transpose(2, 0, 1)).to(
            torch.float32).cuda().unsqueeze(0)

        data['inputs']['query_image'] = query_image

        data = self.model(data)

        lo_ori_reg = data['predictions']['lo_ori_reg']
        lo_ori_cls = data['predictions']['lo_ori_cls']
        lo_centroid = data['predictions']['lo_centroid']
        lo_coeffs = data['predictions']['lo_coeffs']
        pitch_cls = data['predictions']['pitch_cls']
        pitch_reg = data['predictions']['pitch_reg']
        roll_cls = data['predictions']['roll_cls']
        roll_reg = data['predictions']['roll_reg']

        lo_bdb3D_out = get_layout_bdb_sunrgbd(self.bins_tensor, lo_ori_reg,
                                              torch.argmax(lo_ori_cls, 1),
                                              lo_centroid, lo_coeffs)
        lo_bdb3D_out = torch.squeeze(lo_bdb3D_out).detach().cpu().numpy()

        cam_R_out = get_rotation_matix_result(self.bins_tensor,
                                              torch.argmax(pitch_cls,
                                                           1), pitch_reg,
                                              torch.argmax(roll_cls, 1),
                                              roll_reg)
        cam_R_out = cam_R_out.detach().cpu().numpy()

        layout = format_layout(lo_bdb3D_out)

        cam_K = np.loadtxt(
            '../implicit-3d-understanding/demo/inputs/1/cam_K.txt')

        scene_box = Box(resized_image,
                        None,
                        cam_K,
                        None,
                        cam_R_out,
                        None,
                        layout,
                        None,
                        None,
                        'prediction',
                        output_mesh=None)

        scene_box.draw3D(if_save=False, save_path=None)
        return data
