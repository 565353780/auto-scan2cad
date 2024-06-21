#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from udf_generate.Module.udf_generator import UDFGenerator


def demo():
    mesh_file_path = \
        "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/02691156/1a04e3eab45ca15dd86060f189eb133/models/model_normalized.obj"
    udf_save_file_basepath = "/home/chli/chLi/ShapeNet/udfs/02691156/1a04e3eab45ca15dd86060f189eb133/udf"

    udf_generator = UDFGenerator(mesh_file_path)
    udf_generator.generateUDF(udf_save_file_basepath)

    udf_file_name_list = os.listdir(udf_save_file_basepath[:-3])
    print(udf_file_name_list)
    return True
