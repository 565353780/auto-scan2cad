#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from subprocess import Popen, PIPE, STDOUT

from scannet_dataset_manage.Method.path import createFileFolder, renameFile


def runCMD(cmd, print_progress=False):
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=STDOUT)
    if print_progress:
        for line in p.stdout.readlines():
            print(line)
    return p.wait()


class GLBGenerator(object):

    def __init__(self, scannet_dataset_folder_path,
                 scannet_glb_dataset_folder_path):
        self.scannet_dataset_folder_path = scannet_dataset_folder_path
        self.scannet_glb_dataset_folder_path = scannet_glb_dataset_folder_path
        return

    def generateSceneGLB(self, scannet_scene_name):
        scannet_scene_folder_path = \
            self.scannet_dataset_folder_path + scannet_scene_name + "/"

        assert os.path.exists(scannet_scene_folder_path)

        scannet_scene_mesh_file_path = \
            scannet_scene_folder_path + scannet_scene_name + "_vh_clean.ply"
        scannet_scene_glb_file_path = \
            self.scannet_glb_dataset_folder_path + scannet_scene_name + "/" + \
            scannet_scene_name + "_vh_clean.glb"

        if os.path.exists(scannet_scene_glb_file_path):
            return True

        tmp_scannet_scene_glb_file_path = scannet_scene_glb_file_path[:-4] + "_tmp.glb"

        createFileFolder(tmp_scannet_scene_glb_file_path)

        cmd = "assimp export " + \
            scannet_scene_mesh_file_path + \
            " " + tmp_scannet_scene_glb_file_path

        runCMD(cmd)

        renameFile(tmp_scannet_scene_glb_file_path,
                   scannet_scene_glb_file_path)
        return True

    def generateAll(self, print_progress=False):
        scannet_scene_name_list = os.listdir(self.scannet_dataset_folder_path)

        for_data = scannet_scene_name_list
        if print_progress:
            print("[INFO][GLBGenerator::generateAll]")
            print("\t start generate all glb files...")
            for_data = tqdm(for_data)
        for scannet_scene_name in for_data:
            self.generateSceneGLB(scannet_scene_name)
        return True
