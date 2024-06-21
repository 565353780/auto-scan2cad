#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import cv2
import numpy as np

from scene_layout_detect.Method.cluster import clusterPolylines


def getPolylines(explore_map, dist_max=4):
    if len(explore_map.shape) > 2:
        image_gray = cv2.cvtColor(explore_map, cv2.COLOR_RGB2GRAY)
    else:
        image_gray = explore_map

    unknown_mask = (image_gray < 192) & (image_gray > 64)

    thresh = np.ones_like(image_gray, dtype=np.uint8) * 255
    thresh[np.where(unknown_mask)] = 0

    contours, _ = cv2.findContours(thresh, 3, 2)

    max_point_num = 0
    max_point_idx = -1
    for i, contour in enumerate(contours):
        if contour.shape[0] > max_point_num:
            max_point_num = contour.shape[0]
            max_point_idx = i
    cnt = contours[max_point_idx]

    polylines = cv2.approxPolyDP(cnt, dist_max, True)

    polylines = polylines.reshape(-1, 2)[..., ::-1]
    return polylines


global render_idx
render_idx = 1


def getBox(layout_map, render=False):
    bound_mask = np.dstack(np.where(layout_map == 255))[0]
    rect = cv2.minAreaRect(bound_mask)
    box = cv2.boxPoints(rect)

    if render:
        render_image = deepcopy(layout_map)
        draw_box = box[..., ::-1].astype(int)
        cv2.polylines(render_image, [draw_box], True, 128, 1)
        global render_idx
        cv2.imshow("layout_map box " + str(render_idx), render_image)
        render_idx += 1
    return box


def getPolygon(explore_map, dist_max=4, render=False):
    polylines = getPolylines(explore_map, dist_max)
    polygon = clusterPolylines(polylines)

    if render:
        render_image = deepcopy(explore_map)
        if len(render_image.shape) > 2:
            cv2.polylines(render_image,
                          np.int32([polygon])[..., ::-1], True, (0, 0, 255), 1)
        else:
            cv2.polylines(render_image,
                          np.int32([polygon])[..., ::-1], True, 255, 1)
        global render_idx
        cv2.imshow("explore_map polygon " + str(render_idx), render_image)
        render_idx += 1
    return polygon


def getBoundary(explore_map, mode, dist_max=4, render=False):
    mode_list = ['box', 'polygon']

    assert mode in mode_list

    if mode == 'box':
        return getBox(explore_map, render)
    if mode == 'polygon':
        return getPolygon(explore_map, dist_max, render)
