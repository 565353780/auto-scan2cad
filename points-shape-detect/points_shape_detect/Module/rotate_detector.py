#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm

from points_shape_detect.Model.rotate.continus_rotate_net import ContinusRotateNet
from points_shape_detect.Model.rotate.transformer_rotate_net import TransformerRotateNet

from points_shape_detect.Method.sample import fps
from points_shape_detect.Method.device import toCuda


class RotateDetector(object):

    def __init__(self, model_file_path=None):
        #  self.model = ContinusRotateNet(True).cuda()
        self.model = TransformerRotateNet(True).cuda()

        if model_file_path is not None:
            self.loadModel(model_file_path)
        return

    def loadModel(self, model_file_path):
        assert os.path.exists(model_file_path)

        print("[INFO][RotateDetector::loadModel]")
        print("\t start loading model from :")
        print("\t", model_file_path)
        model_dict = torch.load(model_file_path)
        self.model.load_state_dict(model_dict['model'])
        return True

    def detectPointArray(self, point_array):
        self.model.eval()

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        points = torch.from_numpy(point_array.astype(np.float32)).cuda()
        points_center = torch.mean(points, 0)
        origin_points = points - points_center
        origin_points = origin_points.unsqueeze(0)

        fps_points = fps(origin_points, 2048)

        data['inputs']['trans_query_points_center'] = points_center.unsqueeze(
            0)
        data['inputs']['origin_query_point_array'] = fps_points

        data = self.model(data)
        return data
