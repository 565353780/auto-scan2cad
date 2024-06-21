#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
from multiprocessing import Pool

import numpy as np
import open3d as o3d
from conv_onet.Data.crop_space import CropSpace
from points_shape_detect.Method.trans import normalizePointArray
from tqdm import tqdm

from global_to_patch_retrieval.Method.path import createFileFolder, renameFile


def getPointsFeature(point_array, normalize=True):
    crop_space = CropSpace(0.1, 0.1, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])

    if normalize:
        points = normalizePointArray(point_array)
    else:
        points = point_array
    points = points.reshape(1, -1, 3)
    crop_space.updatePointArray(points)

    feature = crop_space.getFeatureArray('valid')
    mask = crop_space.getFeatureMaskArray('valid')
    return feature, mask


def generateCADFeature(model_file_path, shapenet_feature_folder_path):
    assert os.path.exists(model_file_path)
    model_label = model_file_path.split("ShapeNetCore.v2/")[1].split(
        "/models/model_normalized.obj")[0].replace("/", "_")
    feature_file_path = shapenet_feature_folder_path + model_label + ".pkl"

    if os.path.exists(feature_file_path):
        return True

    tmp_feature_file_path = feature_file_path[:-4] + "_tmp.pkl"
    createFileFolder(tmp_feature_file_path)

    mesh = o3d.io.read_triangle_mesh(model_file_path)
    pcd = mesh.sample_points_uniformly(100000)
    points = np.array(pcd.points, dtype=np.float32)

    feature, mask = getPointsFeature(points)

    feature_dict = {'feature': feature, 'mask': mask}
    with open(tmp_feature_file_path, 'wb') as f:
        pickle.dump(feature_dict, f)
    renameFile(tmp_feature_file_path, feature_file_path)
    return True


def generateCADFeatureWithPool(inputs):
    model_file_path, shapenet_feature_folder_path = inputs
    generateCADFeature(model_file_path, shapenet_feature_folder_path)
    return True


def generateAllCADFeatureWithPool(shapenet_model_file_path_list,
                                  shapenet_feature_folder_path,
                                  print_progress=False):
    inputs_list = []
    for shapenet_model_file_path in shapenet_model_file_path_list:
        inputs_list.append(
            [shapenet_model_file_path, shapenet_feature_folder_path])

    pool = Pool(processes=os.cpu_count())
    if print_progress:
        print("[INFO][feature::generateAllCADFeatureWithPool]")
        print("\t start generate shapenet model CAD features with pool...")
        result = list(
            tqdm(pool.imap(generateCADFeatureWithPool, inputs_list),
                 total=len(inputs_list)))
    else:
        result = pool.imap(generateCADFeatureWithPool, inputs_list)
    return True


def generateAllCADFeature(shapenet_model_file_path_list,
                          shapenet_feature_folder_path,
                          print_progress=False,
                          with_pool=False):
    if with_pool:
        return generateAllCADFeatureWithPool(shapenet_model_file_path_list,
                                             shapenet_feature_folder_path,
                                             print_progress)

    for_data = shapenet_model_file_path_list
    if print_progress:
        print("[INFO][feature::generateAllCADFeature]")
        print("\t start generate shapenet model CAD features...")
        for_data = tqdm(for_data)
    for shapenet_model_file_path in for_data:
        generateCADFeature(shapenet_model_file_path,
                           shapenet_feature_folder_path)
    return True
