#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


def getScenePCDList(point_image_list,
                    background_only=False,
                    is_gray=False,
                    translate=[0, 0, 0],
                    estimate_normals=False):
    translate = np.array(translate, dtype=float)

    pcd_list = []

    for point_image in point_image_list:
        point_idx = np.where(point_image.point_array[:, 0] != float("inf"))[0]

        valid_point_idx = []

        points = []
        source_colors = point_image.image.reshape(-1, 3)[..., [2, 1, 0]]
        colors = []

        for i in point_idx:
            label_dict = point_image.label_dict_list[i]
            if "empty" in label_dict.keys():
                continue

            if background_only:
                if "background" not in label_dict.keys():
                    continue

            valid_point_idx.append(i)

            if is_gray:
                source_color = source_colors[i]
                gray_value = 0.3 * source_color[2] + 0.59 * source_color[
                    1] + 0.11 * source_color[0]
                color = [gray_value, gray_value, gray_value]
            else:
                color = source_colors[i]

            colors.append(color)

        points = point_image.point_array[valid_point_idx]
        colors = source_colors[valid_point_idx]
        if is_gray:
            gray_colors = 0.3 * colors[:,
                                       2] + 0.59 * colors[:,
                                                          1] + 0.11 * colors[:,
                                                                             0]
            colors = np.tile(gray_colors.reshape(-1, 1), 3)

        points = np.array(points, dtype=float)
        colors = np.array(colors, dtype=float) / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.translate(translate)

        if estimate_normals:
            pcd.estimate_normals()

        pcd_list.append(pcd)

    return pcd_list


def getMergedScenePCD(point_image_list,
                      background_only=False,
                      is_gray=False,
                      translate=[0, 0, 0],
                      estimate_normals=False):
    scene_pcd_list = getScenePCDList(point_image_list, background_only,
                                     is_gray, translate, estimate_normals)

    points_list = []
    colors_list = []
    for scene_pcd in scene_pcd_list:
        points = np.array(scene_pcd.points)
        colors = np.array(scene_pcd.colors)
        points_list.append(points)
        colors_list.append(colors)

    points = np.vstack(points_list)
    colors = np.vstack(colors_list)
    merged_scene_pcd = o3d.geometry.PointCloud()
    merged_scene_pcd.points = o3d.utility.Vector3dVector(points)
    merged_scene_pcd.colors = o3d.utility.Vector3dVector(colors)
    return merged_scene_pcd
