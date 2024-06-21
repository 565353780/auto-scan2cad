#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shapenet_dataset_manage.Module.Core.V2.model_loader import ModelLoader


def demo():
    model_root_path = \
        "/home/chli/scan2cad/shapenet/ShapeNetCore.v2/02691156/10155655850468db78d106ce0a280f87/"

    model_loader = ModelLoader()
    model_loader.loadModel(model_root_path)
    model_loader.outputInfo()
    return True
