#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../scannet-sim-manage')

import os
import cv2

from scene_layout_detect.Method.boundary import getBoundary


def test():
    image_file_folder = "../auto-cad-recon/output/explore/"
    dist_max = 4
    render = False
    print_progress = True

    image_filename_list = os.listdir(image_file_folder)

    for image_filename in image_filename_list:
        if image_filename[-4:] != ".png":
            continue

        image_file_path = image_file_folder + image_filename

        explore_map = cv2.imread(image_file_path)

        boundary = getBoundary(explore_map, dist_max, render, print_progress)

    cv2.waitKey(0)
    return True
