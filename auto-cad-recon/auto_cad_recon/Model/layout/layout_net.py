#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from auto_cad_recon.Model.layout.resnet import resnet34
from auto_cad_recon.Model.layout.resnet import model_urls


class LayoutNet(nn.Module):

    def __init__(self):
        super().__init__()
        '''Module parameters'''
        self.PITCH_BIN = 2
        self.ROLL_BIN = 2
        self.LO_ORI_BIN = 2
        '''Modules'''
        self.resnet = resnet34(pretrained=False)
        self.fc_1 = nn.Linear(2048, 1024)
        self.fc_2 = nn.Linear(1024, (self.PITCH_BIN + self.ROLL_BIN) * 2)

        # fc for layout
        self.fc_layout = nn.Linear(2048, 2048)
        # for layout orientation
        self.fc_3 = nn.Linear(2048, 1024)
        self.fc_4 = nn.Linear(1024, self.LO_ORI_BIN * 2)
        # for layout centroid and coefficients
        self.fc_5 = nn.Linear(2048, 1024)
        self.fc_6 = nn.Linear(1024, 6)

        self.relu_1 = nn.LeakyReLU(0.2, inplace=True)
        self.dropout_1 = nn.Dropout(p=0.5)

        # initiate weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

        # load pretrained resnet
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = self.resnet.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.resnet.load_state_dict(model_dict)
        return

    def forward(self, data):
        x = data['inputs']['query_image']

        x = self.resnet(x)

        # branch for camera parameters
        cam = self.fc_1(x)
        cam = self.relu_1(cam)
        cam = self.dropout_1(cam)
        cam = self.fc_2(cam)
        pitch_reg = cam[:, 0:self.PITCH_BIN]
        pitch_cls = cam[:, self.PITCH_BIN:self.PITCH_BIN * 2]
        roll_reg = cam[:,
                       self.PITCH_BIN * 2:self.PITCH_BIN * 2 + self.ROLL_BIN]
        roll_cls = cam[:, self.PITCH_BIN * 2 +
                       self.ROLL_BIN:self.PITCH_BIN * 2 + self.ROLL_BIN * 2]

        # branch for layout orientation, centroid and coefficients
        lo = self.fc_layout(x)
        lo = self.relu_1(lo)
        lo = self.dropout_1(lo)
        # branch for layout orientation
        lo_ori = self.fc_3(lo)
        lo_ori = self.relu_1(lo_ori)
        lo_ori = self.dropout_1(lo_ori)
        lo_ori = self.fc_4(lo_ori)
        lo_ori_reg = lo_ori[:, :self.LO_ORI_BIN]
        lo_ori_cls = lo_ori[:, self.LO_ORI_BIN:]

        # branch for layout centroid and coefficients
        lo_ct = self.fc_5(lo)
        lo_ct = self.relu_1(lo_ct)
        lo_ct = self.dropout_1(lo_ct)
        lo_ct = self.fc_6(lo_ct)
        lo_centroid = lo_ct[:, :3]
        lo_coeffs = lo_ct[:, 3:]

        data['predictions']['pitch_reg'] = pitch_reg
        data['predictions']['roll_reg'] = roll_reg
        data['predictions']['pitch_cls'] = pitch_cls
        data['predictions']['roll_cls'] = roll_cls
        data['predictions']['lo_ori_reg'] = lo_ori_reg
        data['predictions']['lo_ori_cls'] = lo_ori_cls
        data['predictions']['lo_centroid'] = lo_centroid
        data['predictions']['lo_coeffs'] = lo_coeffs
        data['predictions']['a_features'] = x
        return data
