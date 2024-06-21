#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

from global_pose_refine.Model.gcnn.collection_unit import CollectionUnit


class GraphConvolutionLayerCollect(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """

    def __init__(self, dim_obj, dim_rel):
        super().__init__()
        self.collect_units = nn.ModuleList()
        # obj (subject) from rel
        self.collect_units.append(CollectionUnit(dim_rel, dim_obj))
        # obj (object) from rel
        self.collect_units.append(CollectionUnit(dim_rel, dim_obj))
        # rel from obj (subject)
        self.collect_units.append(CollectionUnit(dim_obj, dim_rel))
        # rel from obj (object)
        self.collect_units.append(CollectionUnit(dim_obj, dim_rel))
        # obj from obj
        self.collect_units.append(CollectionUnit(dim_obj, dim_obj))
        return

    def forward(self, target, source, attention, unit_id):
        collection = self.collect_units[unit_id](target, source, attention)
        return collection
