#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../scan2cad-dataset-manage")

from auto_cad_recon.Module.renderer import Renderer


def demo():
    data_list_json_folder_path = "./output/20221026_14:41:17/result/scene0474_02/"
    print_progress = True

    renderer = Renderer()
    renderer.renderResult(data_list_json_folder_path, print_progress)
    return True
