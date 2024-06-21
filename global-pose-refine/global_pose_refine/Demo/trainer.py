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

from global_pose_refine.Module.trainer import Trainer


def demo():
    model_file_path = './output/20230423_18:36:38/model_best.pth'
    resume_model_only = False
    print_progress = True

    trainer = Trainer()
    trainer.loadModel(model_file_path, resume_model_only)
    #  trainer.testTrainOnDataset()
    #  trainer.testTrain()
    trainer.train(print_progress)
    return True
