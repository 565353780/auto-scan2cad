#!/usr/bin/env python
# -*- coding: utf-8 -*-

from udf_generate.Module.udf_generate_manager import UDFGenerateManager


def demo():
    mesh_root_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
    udf_save_root_folder_path = "/home/chli/chLi/ShapeNet/udfs/"

    udf_generate_manager = UDFGenerateManager(mesh_root_folder_path)
    udf_generate_manager.activeGenerateAllUDF(udf_save_root_folder_path)
    #  udf_generate_manager.generateAllUDF(udf_save_root_folder_path)
    return True
