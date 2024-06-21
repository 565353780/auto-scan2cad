#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from mesh_manage.Method.trans import transFormat

def getFileFolderPath(file_path):
    file_name = file_path.split("/")[-1]
    file_folder_path = file_path.split(file_name)[0]
    return file_folder_path

def createFileFolder(file_path):
    file_folder_path = getFileFolderPath(file_path)
    os.makedirs(file_folder_path, exist_ok=True)
    return True

def isDataAscii(file_path):
    if not os.path.exists(file_path):
        return False

    with open(file_path, "r") as f:
        lines = []
        try:
            lines = f.readlines()
        except:
            return False

        for line in lines:
            if "DATA" in line:
                if "ascii" not in line:
                    return False
                return True

            if "format" in line:
                if "ascii" not in line:
                    return False
                return True

    print("[ERROR][path::isDataAscii]")
    print("\t not find format!")
    return False

def getValidFilePath(pointcloud_file_path):
    if not os.path.exists(pointcloud_file_path):
        print("[ERROR][path::getValidFilePath]")
        print("\t file not exist!")
        return None

    if isDataAscii(pointcloud_file_path):
        return pointcloud_file_path

    valid_file_path = pointcloud_file_path[:-4] + "_ascii" + pointcloud_file_path[-4:]
    if not os.path.exists(valid_file_path):
        transFormat(pointcloud_file_path, valid_file_path)
    return valid_file_path

