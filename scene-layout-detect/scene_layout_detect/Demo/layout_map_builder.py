#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('../scannet-sim-manage')

import cv2

from scene_layout_detect.Module.layout_map_builder import LayoutMapBuilder


def demo():
    delta_angle = 2
    unit_size = 0.01
    free_width = 50
    wall_height = 3
    explore_paint_radius = 0.1
    min_explore_point_dist = 5
    dist_max = 4
    skip_floor = False
    render = False

    layout_map_builder = LayoutMapBuilder(delta_angle, unit_size, free_width)

    layout_map_builder.addPoints([0, 0, 0], [
        [1, 0, 0],
        [2, 0, 0],
        [2, 1, 0],
        [1, 1, 0],
        [1, 2, 0],
        [0, 2, 0],
        [0, 1, 0],
        [0, 0.5, 0],
    ], explore_paint_radius, render)
    layout_map_builder.updateLayoutMesh(wall_height, dist_max, skip_floor,
                                        render)
    layout_map_builder.updateExplorePointIdx(min_explore_point_dist, render)

    if render:
        cv2.waitKey(0)
    return True
