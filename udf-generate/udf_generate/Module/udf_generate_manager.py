#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from multiprocessing import Pool

from udf_generate.Method.paths import renameFile

from udf_generate.Module.udf_generator import UDFGenerator

# FIXME: only for shapenet core v2
mesh_total_num = 52472

class UDFGenerateManager(object):

    def __init__(self, mesh_root_folder_path, processes=os.cpu_count()):
        self.mesh_root_folder_path = mesh_root_folder_path
        self.processes = processes
        return

    def getMeshFilePathList(self):
        assert os.path.exists(self.mesh_root_folder_path)

        mesh_file_path_list = []

        print("[INFO][UDFGenerateManager::getMeshFilePathList]")
        print("\t start load mesh file path list...")
        pbar = tqdm(total=mesh_total_num)
        for root, _, files in os.walk(self.mesh_root_folder_path):
            for file_name in files:
                if file_name[-4:] != ".obj":
                    continue

                file_path = root + "/" + file_name
                if not os.path.exists(file_path):
                    continue

                mesh_file_path_list.append(file_path)
                pbar.update(1)

        pbar.close()
        return mesh_file_path_list

    def generateSingleUDF(self, inputs):
        '''
        inputs: [mesh_file_path, udf_save_file_basepath]
        '''
        assert len(inputs) == 2

        mesh_file_path, udf_save_file_basepath = inputs

        udf_basename = udf_save_file_basepath.split("/")[-1]
        udf_save_folder_path = udf_save_file_basepath[:-len(udf_basename)]
        if os.path.exists(udf_save_folder_path):
            return True

        tmp_udf_save_folder_path = udf_save_folder_path[:-1] + "_tmp/"
        tmp_udf_save_file_basepath = tmp_udf_save_folder_path + udf_basename

        udf_generator = UDFGenerator(mesh_file_path)
        udf_generator.generateUDF(tmp_udf_save_file_basepath)

        renameFile(tmp_udf_save_folder_path, udf_save_folder_path)
        return True

    def activeGenerateAllUDF(self, udf_save_root_folder_path):
        assert os.path.exists(self.mesh_root_folder_path)

        print("[INFO][UDFGenerateManager::activeGenerateAllUDF]")
        print("\t start load mesh file path and generate udf...")
        pbar = tqdm(total=mesh_total_num)
        for root, _, files in os.walk(self.mesh_root_folder_path):
            for file_name in files:
                if file_name[-4:] != ".obj":
                    continue

                file_path = root + "/" + file_name
                if not os.path.exists(file_path):
                    continue

                mesh_file_basename = file_name[:-4]
                mesh_label_path = file_path.replace(
                    self.mesh_root_folder_path, "").replace(file_name, "")
                udf_save_file_basepath = udf_save_root_folder_path + \
                    mesh_label_path + mesh_file_basename + "/udf"
                inputs = [file_path, udf_save_file_basepath]
                self.generateSingleUDF(inputs)

                pbar.update(1)

        pbar.close()
        return True

    def generateAllUDF(self, udf_save_root_folder_path):
        mesh_file_path_list = self.getMeshFilePathList()

        inputs_list = []
        for mesh_file_path in mesh_file_path_list:
            mesh_file_name = mesh_file_path.split("/")[-1]
            mesh_file_basename = mesh_file_name[:-4]
            mesh_label_path = mesh_file_path.replace(
                self.mesh_root_folder_path, "").replace(mesh_file_name, "")
            udf_save_file_basepath = udf_save_root_folder_path + \
                mesh_label_path + mesh_file_basename + "/udf"
            inputs_list.append([mesh_file_path, udf_save_file_basepath])

        print("[INFO][UDFGenerateManager::generateAllUDF]")
        print("\t start running generateSingleUDF...")
        for inputs in tqdm(inputs_list):
            self.generateSingleUDF(inputs)
        return True

        pool = Pool(processes=self.processes)
        print("[INFO][UDFGenerateManager::generateAllUDF]")
        print("\t start running generateSingleUDF with pool...")
        _ = list(
            tqdm(pool.imap(self.generateSingleUDF, inputs_list),
                 total=len(inputs_list)))
        pool.close()
        pool.join()
        return True
