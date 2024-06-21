#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mesh_manage.Method.trans import transFormat

# Param
source_pointcloud_file_path = \
    "/home/chli/chLi/coscan_data/scene_result/matterport3d_05/dong/2022_9_8_12-29-59_mp3d05_dong/scene_16.pcd"
target_pointcloud_file_path = \
    "/home/chli/chLi/coscan_data/scene_result/matterport3d_05/dong/scene_16.ply"

# Process
transFormat(source_pointcloud_file_path,
            target_pointcloud_file_path,
            True, True)

