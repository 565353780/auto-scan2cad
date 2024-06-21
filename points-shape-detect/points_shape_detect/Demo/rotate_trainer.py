#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../udf-generate")
sys.path.append("../auto-cad-recon")
sys.path.append("../mesh-manage/")
sys.path.append("../scannet-dataset-manage")
sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../shapenet-dataset-manage")

from points_shape_detect.Module.rotate_trainer import RotateTrainer


def demo():
    #  model_file_path = "./output/pretrained_continus_rotate1/model_best.pth"
    model_file_path = "./output/pretrained_transformer_rotate1/model_best.pth"
    resume_model_only = True
    print_progress = True

    rotate_trainer = RotateTrainer()
    rotate_trainer.loadModel(model_file_path, resume_model_only)
    #  rotate_trainer.testTrain()
    rotate_trainer.train(print_progress)
    return True
