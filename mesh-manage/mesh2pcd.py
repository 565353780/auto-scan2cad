#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mesh_manage.Method.sample import samplePointCloudFromMesh

mesh_file_path = "/home/chli/chLi/3.obj"
pcd_basename = "/home/chli/chLi/3_pc_"

sample_num_list = [2048, 4096, 10000, 100000]

for sample_num in sample_num_list:
    samplePointCloudFromMesh(mesh_file_path,
                             pcd_basename + str(sample_num) + ".ply",
                             sample_num)
