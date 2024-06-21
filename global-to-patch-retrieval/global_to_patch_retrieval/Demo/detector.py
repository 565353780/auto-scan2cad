#!/usr/bin/env python
# -*- coding: utf-8 -*-

from global_to_patch_retrieval.Module.detector import Detector


def demo():
    model_file_path = './output/pretrained_retrieval/model_eval_best.pth'

    detector = Detector(model_file_path)
    detector.detectDataset()
    return True
