#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def getFileFolderPath(file_path):
    file_name = file_path.split("/")[-1]
    file_folder_path = file_path.split("/" + file_name)[0] + "/"
    return file_folder_path


def createFileFolder(file_path):
    file_folder_path = getFileFolderPath(file_path)
    os.makedirs(file_folder_path, exist_ok=True)
    return True


def getFilePath(file_basepath, param_name_value_list=[], file_format="npy"):
    file_path = file_basepath
    for param_name_value in param_name_value_list:
        file_path += "_" + param_name_value[0] + "_" + str(param_name_value[1])
    file_path += "." + file_format
    return file_path


def renameFile(source_file_path, target_file_path):
    assert os.path.exists(source_file_path)

    while os.path.exists(source_file_path):
        try:
            os.rename(source_file_path, target_file_path)
        except:
            continue
    return True
