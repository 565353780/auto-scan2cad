#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mesh_manage.Module.renderer import Renderer


def demo():
    for idx in range(28):
    #  for idx in range(27, -1, -1):
        mesh_file_path = \
                "/home/chli/chLi/coscan_data/fast_forward/ff_recon_result_render_gray_blue/merged_vpp_mesh_" + str(idx) + ".ply"
        save_image_file_path = \
                "/home/chli/chLi/coscan_data/fast_forward/ff_recon_result_render_gray_blue_image/merged_vpp_mesh_" + str(idx) + ".png"

        renderer = Renderer()

        print("start render image :", idx + 1, "/", 28)
        #  renderer.renderMesh(mesh_file_path)
        renderer.saveRenderMeshImage(mesh_file_path, save_image_file_path)
    return True
