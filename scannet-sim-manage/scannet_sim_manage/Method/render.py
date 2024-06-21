#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from threading import Thread

from scannet_sim_manage.Method.bbox import getOpen3DBBoxFromBBox


def drawGeometries(geometry_list, window_name="Open3D"):
    thread = Thread(target=o3d.visualization.draw_geometries,
                    args=(geometry_list, window_name))
    thread.start()
    return True


def getBBoxPCDList(bbox_dict):
    bbox_list = []
    for bbox in bbox_dict.values():
        bbox_list.append(getOpen3DBBoxFromBBox(bbox))
    return bbox_list


def getPointImagePCD(point_image):
    points = point_image.point_array[np.where(
        point_image.point_array[:, 0] != float("inf"))[0]]

    colors = point_image.image.reshape(-1, 3)[np.where(
        point_image.point_array[:, 0] != float("inf"))[0]][...,
                                                           [2, 1, 0]] / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def getLabeledPointImagePCD(point_image, label_color_dict={}):
    point_idx = np.where(point_image.point_array[:, 0] != float("inf"))[0]

    points = point_image.point_array[point_idx]
    source_colors = point_image.image.reshape(-1, 3)[..., [2, 1, 0]]
    colors = []

    #  background_color = np.zeros(3)
    background_color = None

    for i in point_idx:
        label_dict = point_image.label_dict_list[i]
        if "background" in label_dict.keys():
            if background_color is None:
                #  colors.append(source_colors[i])

                source_color = source_colors[i]
                gray_value = 0.3 * source_color[2] + 0.59 * source_color[
                    1] + 0.11 * source_color[0]
                gray_color = [gray_value, gray_value, gray_value]
                colors.append(gray_color)
            else:
                colors.append(background_color)
            continue

        assert "empty" not in label_dict.keys()

        for key in label_dict.keys():
            if key in label_color_dict.keys():
                continue

            new_color = np.random.rand(3) * 255.0
            label_color_dict[key] = new_color

        label = list(label_dict.keys())[0]
        color = label_color_dict[label]
        colors.append(color)

    colors = np.array(colors, dtype=float) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd, label_color_dict


def isAnyPointInBBox(bbox, point_array):
    if len(point_array) == 0:
        return False

    mask_x = (point_array[:, 0] >= bbox.min_point.x) &\
        (point_array[:, 0] <= bbox.max_point.x)
    if True not in mask_x:
        return False

    mask_y = (point_array[:, 1] >= bbox.min_point.y) &\
        (point_array[:, 1] <= bbox.max_point.y)
    if True not in mask_y:
        return False

    mask_z = (point_array[:, 2] >= bbox.min_point.z) &\
        (point_array[:, 2] <= bbox.max_point.z)
    if True not in mask_z:
        return False

    mask = mask_x & mask_y & mask_z
    return True in mask


def getValidBBoxPCDList(bbox_dict, point_image):
    points = point_image.point_array[np.where(
        point_image.point_array[:, 0] != float("inf"))[0]]

    bbox_list = []
    for bbox in bbox_dict.values():
        if not isAnyPointInBBox(bbox, points):
            continue
        bbox_list.append(getOpen3DBBoxFromBBox(bbox))
    return bbox_list


def renderBBox(bbox_dict):
    bbox_list = getBBoxPCDList(bbox_dict)
    drawGeometries(bbox_list)
    return True


def renderPointImage(point_image):
    pcd = getPointImagePCD(point_image)
    drawGeometries([pcd])
    return True


def renderLabeledPointImage(point_image, label_color_dict, render=True):
    pcd, label_color_dict = getLabeledPointImagePCD(point_image,
                                                    label_color_dict)
    if render:
        drawGeometries([pcd])
    return label_color_dict


def renderAll(point_image, bbox_dict):
    pcd = getPointImagePCD(point_image)

    valid_bbox_list = getValidBBoxPCDList(bbox_dict, point_image)

    drawGeometries([pcd] + valid_bbox_list)
    return True
