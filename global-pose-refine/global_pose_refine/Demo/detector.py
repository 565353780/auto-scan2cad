#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append("../points-shape-detect")

from global_pose_refine.Module.detector import Detector


def demo():
    model_file_path = None

    detector = Detector(model_file_path)

    detector.detectSceneTrans(None)
    return True
