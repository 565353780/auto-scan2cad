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


def renameFile(source_file_path, target_file_path):
    assert not os.path.exists(target_file_path)

    while os.path.exists(source_file_path):
        try:
            os.rename(source_file_path, target_file_path)
        except:
            pass
    return True


def removeIfExist(file_path):
    while os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            continue
    return True
