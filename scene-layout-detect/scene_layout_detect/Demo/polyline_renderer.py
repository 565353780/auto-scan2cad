#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scene_layout_detect.Module.polyline_renderer import PolylineRenderer


def demo():
    polyline = [[0, 0], [2, 0], [4, 0], [4, 4], [0, 4]]
    render_mode = 'source+cluster'
    width = 4000
    height = 4000
    free_width = 50
    render_width = 2560
    render_height = 1440
    debug = True
    line_width = 3
    text_color = [0, 0, 255]
    text_size = 1
    text_line_width = 1
    cluster_idx_list = [0, 0, 1, 2, 3]
    wait_key = 0
    window_name = '[Renderer][' + render_mode + ']'

    polyline_renderer = PolylineRenderer(width, height, free_width,
                                         render_width, render_height, debug)
    polyline_renderer.render(polyline, render_mode, line_width, text_color,
                             text_size, text_line_width, cluster_idx_list)
    polyline_renderer.show(wait_key, window_name)
    return True
