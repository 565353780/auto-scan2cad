#!/usr/bin/env python
# -*- coding: utf-8 -*-

from auto_cad_recon.Demo.dataset_manager import demo as demo_manage_dataset
from auto_cad_recon.Demo.dataset_render_manager import demo as demo_render_dataset
from auto_cad_recon.Demo.retrieval_trainer import demo as demo_train_roca_retrieval
from auto_cad_recon.Demo.renderer import demo as demo_render_result
from auto_cad_recon.Demo.retrieval_manager import (
    demo as demo_manage_retrieval,
    demo_render as demo_render_retrieval_result,
)
from auto_cad_recon.Demo.layout_detector import demo as demo_detect_layout
from auto_cad_recon.Demo.nbv_generator import demo as demo_generate_nbv
from auto_cad_recon.Demo.simple_scene_explorer import demo as demo_simple_explore_scene
from auto_cad_recon.Demo.auto_cad_reconstructor import demo as demo_auto_reconstruct_cad

if __name__ == "__main__":
    # module test
    #  demo_manage_dataset()
    #  demo_render_dataset()
    #  demo_train_roca_retrieval()
    #  demo_render_result()
    #  demo_manage_retrieval()
    #  demo_render_retrieval_result()
    #  demo_detect_layout()
    #  demo_generate_nbv()
    #  exit()

    # scene check
    # demo_simple_explore_scene()

    # main loop
    demo_auto_reconstruct_cad()
