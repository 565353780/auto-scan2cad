#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm

from scannet_dataset_manage.Module.sens_reader import SensReader


def demo():
    sens_file_path = '/home/chli/chLi/ScanNet/scans/scene0000_01/scene0000_01.sens'
    sens_file_path = '/home/chli/scene0000_01.sens'
    header_only = False
    output_path = './output/'

    sens_reader = SensReader(sens_file_path, header_only)

    print('num_frames =', sens_reader.num_frames)

    if not header_only:
        sens_reader.export_depth_images(output_path + 'depth/')
        sens_reader.export_color_images(output_path + 'color/')
        sens_reader.export_poses(output_path + 'pose/')
        sens_reader.export_intrinsics(output_path + 'intrinsic/')
    return True


def demo_dataset():
    scannet_dataset_folder_path = '/home/chli/chLi/ScanNet/scans/'
    header_only = True

    num_frame_list = []

    scene_foldername_list = os.listdir(scannet_dataset_folder_path)

    for scene_folername in tqdm(scene_foldername_list):
        sens_file_path = scannet_dataset_folder_path + scene_folername + '/' + scene_folername + '.sens'

        if not os.path.exists(sens_file_path):
            continue

        num_frame = SensReader(sens_file_path, header_only).num_frames
        num_frame_list.append(num_frame)

    num_frame_mean = np.mean(num_frame_list)
    print('num_frame_mean =', num_frame_mean)
    num_frame_min = np.min(num_frame_list)
    print('num_frame_min =', num_frame_min)
    num_frame_max = np.max(num_frame_list)
    print('num_frame_max =', num_frame_max)
    return True
