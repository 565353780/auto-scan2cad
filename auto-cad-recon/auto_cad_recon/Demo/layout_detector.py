#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../implicit-3d-understanding")

import cv2

from auto_cad_recon.Module.layout_detector import LayoutDetector


def demo():
    model_file_path = "../implicit-3d-understanding/out/pose_net.pth"
    layout_detector = LayoutDetector(model_file_path)

    image_file_path = "../implicit-3d-understanding/demo/inputs/1/img.jpg"

    image = cv2.imread(image_file_path)

    layout_detector.detectImage(image)
    return True
