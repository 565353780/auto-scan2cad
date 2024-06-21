#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from auto_cad_recon.Model.retrieval.resnet_decoder import ResNetDecoder
from auto_cad_recon.Model.retrieval.resnet_encoder import ResNetEncoder


class RetrievalHead(nn.Module):

    def __init__(self, shape_code_size=512, margin=0.5):
        super().__init__()
        self.shape_code_size = shape_code_size

        self.triplet_loss = nn.TripletMarginLoss(margin=margin,
                                                 reduction='none')

        self.shape_encoder = ResNetEncoder()
        self.shape_decoder = ResNetDecoder(relu_in=True,
                                           feats=self.shape_encoder.feats)

        #  source is nn.BCELoss()
        self.decode_loss = nn.BCEWithLogitsLoss()

        self.scannet_scene_object_encode_dict = {}

        self.param_updated = True
        return

    def resetSceneEncode(self):
        self.param_updated = True
        return True

    def encodeUDF(self, udf):
        return self.shape_encoder(udf)

    def decodeShapeCode(self, shape_code):
        return self.shape_decoder(shape_code)

    def forwardScene(self,
                     scannet_scene_name,
                     dataset_manager,
                     print_progress=False):
        self.scannet_scene_object_encode_dict[scannet_scene_name] = {}

        scannet_object_file_name_list = dataset_manager.getScanNetObjectFileNameList(
            scannet_scene_name)

        for_data = scannet_object_file_name_list
        if print_progress:
            print("[INFO][RetrievalHead::forwardScene]")
            print("\t start forward all scene objects...")
            for_data = tqdm(for_data)
        for scannet_object_file_name in for_data:
            shapenet_model_tensor_dict = dataset_manager.getShapeNetModelTensorDict(
                scannet_scene_name, scannet_object_file_name)
            shapenet_cad_udf = shapenet_model_tensor_dict['cad_udf'].reshape(
                -1, 1, 32, 32, 32).cuda()

            self.scannet_scene_object_encode_dict[scannet_scene_name][
                scannet_object_file_name] = [
                    shapenet_model_tensor_dict['shapenet_model_file_path'],
                    self.encodeUDF(shapenet_cad_udf)
                ]
        return True

    def retrieval_single_loss(self, data, batch_idx):
        scannet_scene_name = data['inputs']['scannet_scene_name']

        if batch_idx == -1:
            scannet_object_file_name = data['inputs'][
                'scannet_object_file_name']
            cad_udf = data['predictions']['cad_udf']
            retrieval_shape_code = data['predictions']['retrieval_shape_code']
        else:
            scannet_object_file_name = data['inputs'][
                'scannet_object_file_name'][batch_idx]
            cad_udf = data['predictions']['cad_udf'][batch_idx].reshape(
                -1, 32, 32, 32)
            retrieval_shape_code = data['predictions']['retrieval_shape_code'][
                batch_idx]

        shape_code = self.scannet_scene_object_encode_dict[scannet_scene_name][
            scannet_object_file_name][1]

        decode_udf = self.decodeShapeCode(shape_code).reshape(-1, 32, 32, 32)

        if 'loss_noc_decode' not in data['losses'].keys():
            data['losses']['loss_noc_decode'] = self.decode_loss(
                decode_udf, cad_udf.detach()).reshape(1)
        else:
            data['losses']['loss_noc_decode'] = torch.cat(
                (data['losses']['loss_noc_decode'],
                 self.decode_loss(decode_udf, cad_udf.detach()).reshape(1)), 0)

        neg_shape_code = [
            self.scannet_scene_object_encode_dict[data['inputs']
                                                  ['scannet_scene_name']]
            [neg_scannet_object_file_name][1].clone()
            for neg_scannet_object_file_name in list(
                self.scannet_scene_object_encode_dict[scannet_scene_name].keys(
                ))
            if neg_scannet_object_file_name.split(".")[0].split(
                "_")[1] != scannet_object_file_name.split(".")[0].split("_")[1]
        ]

        if len(neg_shape_code) == 0:
            return data

        neg = torch.stack(neg_shape_code, 0).reshape(-1, 256)

        anchor = torch.stack(
            [retrieval_shape_code.clone() for _ in range(neg.shape[0])],
            0).reshape(-1, 256)

        pos_shape_code = [
            self.scannet_scene_object_encode_dict[scannet_scene_name]
            [scannet_object_file_name][1].clone() for _ in range(neg.shape[0])
        ]

        pos = torch.stack(pos_shape_code, 0).reshape(-1, 256)

        if 'loss_triplet' not in data['losses'].keys():
            data['losses']['loss_triplet'] = self.triplet_loss(
                anchor, pos, neg)
        else:
            data['losses']['loss_triplet'] = torch.cat(
                (data['losses']['loss_triplet'],
                 self.triplet_loss(anchor, pos, neg)), 0)
        return data

    def retrieval_loss(self, data):
        if isinstance(data['inputs']['scannet_object_file_name'], str):
            data = self.retrieval_single_loss(data, -1)
        else:
            for i in range(len(data['inputs']['scannet_object_file_name'])):
                data = self.retrieval_single_loss(data, i)
        return data

    def forward(self, data, print_progress=False):
        if self.param_updated:
            self.forwardScene(data['inputs']['scannet_scene_name'],
                              data['inputs']['dataset_manager'],
                              print_progress)

        data['predictions']['retrieval_shape_code'] = self.encodeUDF(
            data['predictions']['point_udf'].reshape(-1, 1, 32, 32, 32))

        if isinstance(data['inputs']['scannet_object_file_name'], str):
            min_dist = float('inf')
            min_dist_shapenet_model_file_path = None
            for shapenet_model_file_path, shapenet_model_shape_code in \
                    self.scannet_scene_object_encode_dict[data['inputs']['scannet_scene_name']].values():
                current_dist = F.pairwise_distance(
                    data['predictions']['retrieval_shape_code'],
                    shapenet_model_shape_code)
                if current_dist < min_dist:
                    min_dist = current_dist
                    min_dist_shapenet_model_file_path = shapenet_model_file_path
            data['predictions'][
                'retrieval_model_file_path'] = min_dist_shapenet_model_file_path
        else:
            data['predictions']['retrieval_model_file_path'] = []
            for scannet_object_shape_code in data['predictions'][
                    'retrieval_shape_code']:
                min_dist = float('inf')
                min_dist_shapenet_model_file_path = None
                for shapenet_model_file_path, shapenet_model_shape_code in \
                        self.scannet_scene_object_encode_dict[data['inputs']['scannet_scene_name']].values():
                    current_dist = F.pairwise_distance(
                        scannet_object_shape_code.reshape(1, -1), shapenet_model_shape_code)
                    if current_dist < min_dist:
                        min_dist = current_dist
                        min_dist_shapenet_model_file_path = shapenet_model_file_path
                data['predictions']['retrieval_model_file_path'].append(
                    min_dist_shapenet_model_file_path)

        if self.training:
            data = self.retrieval_loss(data)
        return data
