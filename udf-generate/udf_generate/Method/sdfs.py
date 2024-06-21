#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import trimesh
from mesh_to_sdf import mesh_to_voxels

from udf_generate.Method.paths import createFileFolder


def generateSDF(mesh, resolution):
    sdf = mesh_to_voxels(mesh, resolution, pad=True)
    return sdf


def generateSDFFromFile(mesh_file_path, resolution):
    if not os.path.exists(mesh_file_path):
        print("[ERROR][sdfs::generateSDFFromFile]")
        print("\t mesh_file not exist!")
        return None

    mesh = trimesh.load(mesh_file_path)
    return generateSDF(mesh, resolution)


def saveSDF(sdf, save_file_path):
    if not createFileFolder(save_file_path):
        print("[ERROR][sdfs::saveSDF]")
        print("\t createFileFolder failed!")
        return False

    print("sdf.shape =", sdf.shape)
    print(sdf[0, :10])
    return True
