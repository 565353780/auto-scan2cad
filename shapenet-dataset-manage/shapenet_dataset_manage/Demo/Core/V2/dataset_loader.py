#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from shapenet_dataset_manage.Module.Core.V2.dataset_loader import DatasetLoader


def demo():
    dataset_root_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
    csv_file_path = "/home/chli/chLi/ShapeNet/all.csv"
    msh2df_path = "/home/chli/github/local-deep-implicit-functions/ldif/gaps/bin/x86_64/msh2df"
    grd2msh_path = "/home/chli/github/local-deep-implicit-functions/ldif/gaps/bin/x86_64/grd2msh"
    grd_save_folder_path = "/home/chli/ShapeNet/v2_grd/"
    ply_save_folder_path = "/home/chli/ShapeNet/v2_ply/"
    split_save_folder_path = "/home/chli/ShapeNet/v2_split/"

    dataset_loader = DatasetLoader()
    dataset_loader.loadDataset(dataset_root_path)
    dataset_loader.loadCSVDict(csv_file_path)

    # trans obj to ply
    if False:
        dataset_loader.transObjToPly(msh2df_path, grd2msh_path,
                                     grd_save_folder_path,
                                     ply_save_folder_path)
        dataset_loader.splitPly(ply_save_folder_path, split_save_folder_path)

    # generate mini dataset
    if True:
        mini_dataset_folder_path = "/home/chli/chLi/ShapeNet/mini/"
        os.makedirs(mini_dataset_folder_path, exist_ok=True)

        for synset_id, synset in dataset_loader.dataset.synset_dict.items():
            synset_folder_path = mini_dataset_folder_path + synset_id + "/"
            for model_id, model in synset.model_dict.items():
                model_file_path = model.normalized_obj_file_path
                save_model_file_path = synset_folder_path + model_id + ".ply"
                print(save_model_file_path)
                exit()

    dataset_loader.outputInfo()
    return True
