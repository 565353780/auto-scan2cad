#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
from shutil import copyfile

dataset_root_path = "/home/chli/chLi/Dataset/ShapeNet/Core/ShapeNetCore.v2/"
mini_dataset_root_path = "/home/chli/chLi/Dataset/ShapeNet/mini/"
class_num = -1
model_num = 100

target_class = ['02691156', '03001627']

os.makedirs(mini_dataset_root_path, exist_ok=True)

class_name_list = os.listdir(dataset_root_path)

for class_idx, class_name in enumerate(class_name_list):
    if class_idx == class_num:
        break

    class_folder_path = dataset_root_path + class_name + "/"
    if not os.path.isdir(class_folder_path):
        continue

    if class_name not in target_class:
        continue

    save_class_folder_path = mini_dataset_root_path + class_name + "/"
    os.makedirs(save_class_folder_path, exist_ok=True)

    model_id_list = os.listdir(class_folder_path)

    print("[INFO][create_mini_dataset]")
    print("\t start create for class " + str(class_idx + 1) + "/" +
          str(len(class_name_list)) + "...")

    model_idx = 0
    for model_id in tqdm(model_id_list):
        if model_idx == model_num:
            break

        model_file_path = class_folder_path + model_id + "/models/model_normalized.obj"
        if not os.path.exists(model_file_path):
            continue

        save_model_file_path = save_class_folder_path + model_id + ".obj"

        copyfile(model_file_path, save_model_file_path)
        model_idx += 1
