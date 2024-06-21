#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../udf-generate")
sys.path.append("../auto-cad-recon")
sys.path.append("../mesh-manage/")
sys.path.append("../scannet-dataset-manage")
sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../shapenet-dataset-manage")
sys.path.append("../points-shape-detect")
sys.path.append("../scene-layout-detect")
sys.path.append("../scannet-sim-manage")

from global_pose_refine.Dataset.object_position_dataset import ObjectPositionDataset


def test():
    training = True
    training_percent = 0.8
    object_position_dataset = ObjectPositionDataset(training, training_percent)

    for i in range(len(object_position_dataset)):
        data = object_position_dataset.__getitem__(i)
        if i > 10:
            break
    return True
