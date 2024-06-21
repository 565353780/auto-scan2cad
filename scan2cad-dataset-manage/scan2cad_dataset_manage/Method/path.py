#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

def createFileFolder(file_path):
    file_name = file_path.split("/")[-1]
    file_folder_path = file_path.split(file_name)[0]
    os.makedirs(file_folder_path, exist_ok=True)
    return True

def renameFile(file_path, target_file_path):
    if not os.path.exists(file_path):
        print("[ERROR][path::renameFile]")
        print("\t file not exist!")
        return False

    while os.path.exists(file_path):
        try:
            os.rename(file_path, target_file_path)
        except:
            continue
    return True

def removeFile(file_path):
    while os.path.exists(file_path):
        try:
            os.remove(file_path)
        except:
            continue
    return True

