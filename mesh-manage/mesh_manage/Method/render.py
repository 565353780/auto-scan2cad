#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d

def render(pointcloud_file_path, estimate_normals_radius, estimate_normals_max_nn):
    pointcloud = o3d.io.read_point_cloud(pointcloud_file_path, print_progress=True)
    pointcloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=estimate_normals_radius,
            max_nn=estimate_normals_max_nn))
    o3d.visualization.draw_geometries([pointcloud])
    return True

def Open3DVisualizer(geometry_list):
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(window_name="test")

    view_control = visualizer.get_view_control()

    render_option = visualizer.get_render_option()
    render_option.line_width = 1.0
    render_option.point_size = 10.0
    render_option.background_color = np.array([255.0, 255.0, 255.0])/255.0

    for geometry_object in geometry_list:
        visualizer.add_geometry(geometry_object)

    visualizer.run()
    visualizer.destroy_window()
    return True

