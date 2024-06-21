#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import open3d as o3d
from math import ceil
from copy import deepcopy

from scene_layout_detect.Config.color import (FREE_COLOR, OBSTACLE_COLOR,
                                              UNKNOWN_COLOR)
from scene_layout_detect.Method.render import drawMeshList


class ExploreMap(object):

    def __init__(self,
                 unit_size=0.1,
                 free_width=5,
                 floor_height_max=0.4,
                 agent_height_max=2):
        self.unit_size = unit_size
        self.free_width = free_width
        self.floor_height_max = floor_height_max
        self.agent_height_max = agent_height_max

        self.floor_z_value = None
        self.map_start_point = None
        self.map = None

        self.map_idx = 0
        return

    def reset(self):
        self.map_start_point = None
        self.map = None
        return True

    def updateMap(self, polygon):
        if polygon.shape[0] == 0:
            print('[WARN][ExploreMap::updateMap]')
            print('\t polygon is empty!')
            return True

        scale = 1.0 / self.unit_size

        points = deepcopy(polygon)
        points = points * scale

        min_point = np.min(points, axis=0)
        max_point = np.max(points, axis=0)

        int_min_point = np.array([
            int(min_point[0]) - self.free_width,
            int(min_point[1]) - self.free_width, 0
        ],
                                 dtype=int)

        ceil_max_point = np.array([
            ceil(max_point[0]) + self.free_width,
            ceil(max_point[1]) + self.free_width, 0
        ],
                                  dtype=int)

        map_shape = ceil_max_point - int_min_point + [1, 1, 0]

        if self.map_start_point is None:
            self.map_start_point = int_min_point
            self.map = np.ones(
                (map_shape[0], map_shape[1]), dtype=np.uint8) * UNKNOWN_COLOR
            return True

        new_map_start_point = np.min([self.map_start_point, int_min_point],
                                     axis=0)

        exist_map_start_pixel = self.map_start_point - new_map_start_point

        map_width, map_height = self.map.shape
        new_map_width = max(map_width + exist_map_start_pixel[0],
                            ceil_max_point[0] - new_map_start_point[0] + 1)
        new_map_height = max(map_height + exist_map_start_pixel[1],
                             ceil_max_point[1] - new_map_start_point[1] + 1)

        if (new_map_start_point == self.map_start_point).all() and \
                new_map_width == map_width and \
                new_map_height == map_height:
            return True

        new_map = np.ones(
            (new_map_width, new_map_height), dtype=np.uint8) * UNKNOWN_COLOR

        new_map[exist_map_start_pixel[0]:exist_map_start_pixel[0] + map_width,
                exist_map_start_pixel[1]:exist_map_start_pixel[1] +
                map_height] = self.map

        self.map_start_point = new_map_start_point
        self.map = new_map
        return True

    def getPixelFromPoint(self, point):
        scale_point = point / self.unit_size
        diff = scale_point - self.map_start_point
        pixel = np.array([int(diff[0]), int(diff[1])], dtype=int)
        return pixel

    def getPointFromPixel(self, x, y):
        translate_x = x + self.map_start_point[0]
        translate_y = y + self.map_start_point[1]
        point = np.array(
            [translate_x * self.unit_size, translate_y * self.unit_size, 0],
            dtype=float)
        return point

    def updateFree(self, polygon):
        self.updateMap(polygon)

        points = []
        for point in polygon:
            pixel = self.getPixelFromPoint(point)
            points.append(pixel)
        points = np.array(points, dtype=int)

        new_map = np.zeros_like(self.map, dtype=np.uint8)
        pts = points[..., ::-1]
        cv2.fillPoly(new_map, [pts], 255)

        not_obstacle_mask = self.map != OBSTACLE_COLOR
        free_mask = new_map == 255
        update_free_mask = free_mask & not_obstacle_mask
        update_free_idx = np.where(update_free_mask == True)
        self.map[update_free_idx] = FREE_COLOR
        return True

    def updateObstacle(self, point_array, paint_radius=0.01):
        paint_pixel_length = ceil(paint_radius / self.unit_size)

        for point in point_array:
            z_value = point[2]
            pixel = self.getPixelFromPoint(point)
            x_start_pixel = max(0, pixel[0] - paint_pixel_length)
            x_end_pixel = min(self.map.shape[0], pixel[0] + paint_pixel_length)
            y_start_pixel = max(0, pixel[1] - paint_pixel_length)
            y_end_pixel = min(self.map.shape[1], pixel[1] + paint_pixel_length)
            point_height = z_value - self.floor_z_value
            if point_height < self.floor_height_max or \
                    point_height > self.agent_height_max:
                continue
            self.map[x_start_pixel:x_end_pixel,
                     y_start_pixel:y_end_pixel] = OBSTACLE_COLOR
        return True

    def addPoints(self, point_array, paint_radius=0.1, render=False):
        bound_points = point_array[np.where(
            point_array[:, 0] != float("inf"))[0]]

        min_z_value = np.min(bound_points[:, 2])
        if self.floor_z_value is None:
            self.floor_z_value = min_z_value
        else:
            self.floor_z_value = min(self.floor_z_value, min_z_value)

        self.updateObstacle(bound_points, paint_radius)

        if render:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(bound_points)
            delta_height = self.floor_height_max
            colors = []
            min_z_value = np.min(bound_points[:, 2])
            for point in bound_points:
                z_diff = point[2] - min_z_value
                color_idx = int(z_diff / delta_height) % 3
                color = [0, 0, 0]
                color[color_idx] = 1
                colors.append(color)
            colors = np.array(colors)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            drawMeshList([pcd], "height split color")

        if render:
            cv2.imshow("explore_map", self.map)
            cv2.waitKey(1)

        if False:
            os.makedirs("./output/explore/", exist_ok=True)
            cv2.imwrite("./output/explore/" + str(self.map_idx) + ".png",
                        self.map)
            self.map_idx += 1
        return True
