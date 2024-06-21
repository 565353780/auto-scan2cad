#!/usr/bin/env python
# -*- coding: utf-8 -*-

from global_to_patch_retrieval.Module.trainer import Trainer


def demo():
    model_file_path = './output/pretrained_retrieval/model_best.pth'
    resume_model_only = True
    print_progress = True

    trainer = Trainer()
    trainer.loadModel(model_file_path, resume_model_only)
    #  trainer.testTrain()
    trainer.train(print_progress)
    return True
