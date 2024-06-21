#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn

from global_to_patch_retrieval.Model.global_retrieval.completion_net import CompletionNet
from global_to_patch_retrieval.Model.global_retrieval.separation_net import SeparationNet
from global_to_patch_retrieval.Model.global_retrieval.triplet_net import TripletNet
from global_to_patch_retrieval.Method.weight import setWeight


class RetrievalNet(nn.Module):

    def __init__(self, infer=False):
        super().__init__()

        self.separation_net = SeparationNet()
        self.completion_net = CompletionNet()
        self.triplet_net = TripletNet()

        self.loss_separation = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_completion = nn.BCEWithLogitsLoss(reduction="none")
        self.loss_triplet = nn.TripletMarginLoss(reduction="none", margin=1e-2)

        self.infer = infer
        return

    def separateForeground(self, data):
        scan_model = data['inputs']['scan_content']

        foreground, background = self.separation_net(torch.sigmoid(scan_model))

        data['predictions']['foreground'] = foreground
        data['predictions']['background'] = background

        if not self.infer:
            data = self.lossSeparate(data)
        return data

    def lossSeparate(self, data):
        foreground = data['predictions']['foreground']
        background = data['predictions']['background']
        gt_foreground = data['inputs']['scan_mask']
        scan_model = data['inputs']['scan_content']

        device = foreground.device

        gt_background = torch.where(
            gt_foreground == 0, scan_model,
            torch.zeros(gt_foreground.shape).to(device))

        loss_foreground = torch.mean(self.loss_separation(
            foreground, gt_foreground),
                                     dim=[1, 2, 3, 4]).mean()
        loss_background = torch.mean(self.loss_separation(
            background, gt_background),
                                     dim=[1, 2, 3, 4]).mean()

        data['losses']['loss_foreground'] = loss_foreground
        data['losses']['loss_background'] = loss_background
        return data

    def completeShape(self, data):
        foreground = data['predictions']['foreground']

        completed_shape = self.completion_net(torch.sigmoid(foreground))

        data['predictions']['completed_shape'] = completed_shape

        if not self.infer:
            data = self.lossComplete(data)
        return data

    def lossComplete(self, data):
        completed_shape = data['predictions']['completed_shape']
        gt_completed_shape = data['inputs']['cad_content']

        loss_completion = torch.mean(self.loss_completion(
            completed_shape, gt_completed_shape),
                                     dim=[1, 2, 3, 4]).mean()

        data['losses']['loss_completion'] = loss_completion
        return data

    def clusterFeature(self, data):
        if self.infer:
            if 'completed_shape' in data['predictions'].keys():
                embed_shape = data['predictions']['completed_shape']
            else:
                embed_shape = data['inputs']['scan_content']

            embed_feature = self.triplet_net.embed(embed_shape)

            data['predictions']['embed_feature'] = embed_feature
        else:
            cad_model = data['inputs']['cad_content']
            negative_model = data['inputs']['negative_content']
            completed_shape = data['predictions']['completed_shape']

            anchor, positive, negative = self.triplet_net(
                torch.sigmoid(completed_shape), cad_model, negative_model)

            data['predictions']['anchor'] = anchor
            data['predictions']['positive'] = positive
            data['predictions']['negative'] = negative

            data = self.lossCluster(data)
        return data

    def lossCluster(self, data):
        anchor = data['predictions']['anchor']
        positive = data['predictions']['positive']
        negative = data['predictions']['negative']

        loss_cluster = self.loss_triplet(anchor.view(anchor.shape[0], -1),
                                         positive.view(anchor.shape[0], -1),
                                         negative.view(anchor.shape[0],
                                                       -1)).mean()

        data['losses']['loss_cluster'] = loss_cluster
        return data

    def setWeight(self, data):
        if self.infer:
            return data

        data = setWeight(data, 'loss_foreground', 1)
        data = setWeight(data, 'loss_background', 1)
        data = setWeight(data, 'loss_completion', 1)
        data = setWeight(data, 'loss_cluster', 1)
        return data

    def forward(self, data):
        data = self.separateForeground(data)

        data = self.completeShape(data)

        data = self.clusterFeature(data)

        data = self.setWeight(data)
        return data

    def embedCAD(self, data):
        data = self.clusterFeature(data)
        return data
