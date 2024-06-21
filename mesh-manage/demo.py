#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mesh_manage.Module.painter import demo_merge, demo_paint, demo_auto_paint, demo_merge_with_color
from mesh_manage.Module.sampler import demo_sample_pointcloud, demo_sample_mesh
from mesh_manage.Module.channel_mesh import demo as demo_generate_mesh

from mesh_manage.Demo.renderer import demo as demo_render

if __name__ == "__main__":
    #  demo_generate_mesh()
    #  demo_merge_with_color()
    #  demo_sample_mesh()
    demo_render()
