#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

from global_pose_refine.Model.gcnn.update_unit import UpdateUnit


class GraphConvolutionLayerUpdate(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """

    def __init__(self, dim_obj, dim_rel):
        super().__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(UpdateUnit(dim_obj))  # obj from others
        self.update_units.append(UpdateUnit(dim_rel))  # rel from others
        return

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update
