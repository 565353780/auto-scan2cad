#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from math import ceil

import cv2
import numpy as np

from scannet_sim_manage.Method.render import drawGeometries

from scene_layout_detect.Method.mesh import generateLayoutMesh
from scene_layout_detect.Method.boundary import getBoundary
from scene_layout_detect.Method.render import renderPolygonAndFloor


class LayoutMap(object):
    def __init__(self, unit_size=0.01, free_width=50):
        self.unit_size = unit_size
        self.free_width = free_width

        self.polygon_list = []
        self.map_start_point = None
        self.floor_array = []
        self.map = None
        return

    def reset(self):
        self.polygon_list = []
        self.map_start_point = None
        self.floor_array = []
        self.map = None
        return True

    def addPolygon(self, polygon):
        self.polygon_list.append(deepcopy(polygon))
        return True

    def updateMap(self, unit_size=0.01, free_width=50, fill=False):
        self.unit_size = unit_size
        self.free_width = free_width
        scale = 1.0 / unit_size

        self.map_start_point = None
        self.map = None

        if len(self.polygon_list) == 0:
            return True

        copy_polygon_list = deepcopy(self.polygon_list)

        for i in range(len(copy_polygon_list)):
            copy_polygon_list[i] = copy_polygon_list[i] * scale

        map_x_min = float('inf')
        map_y_min = float('inf')

        for polygon in copy_polygon_list:
            x_min = np.min(polygon[:, 0])
            y_min = np.min(polygon[:, 1])

            map_x_min = min(map_x_min, x_min)
            map_y_min = min(map_y_min, y_min)

        self.map_start_point = np.array([
            int(map_x_min) - self.free_width,
            int(map_y_min) - self.free_width, 0
        ],
                                        dtype=int)

        for i in range(len(copy_polygon_list)):
            copy_polygon_list[i] = copy_polygon_list[i] - self.map_start_point

        map_x_max = -float('inf')
        map_y_max = -float('inf')

        for polygon in copy_polygon_list:
            x_max = np.max(polygon[:, 0])
            y_max = np.max(polygon[:, 1])

            map_x_max = max(map_x_max, x_max)
            map_y_max = max(map_y_max, y_max)

        map_width = ceil(map_x_max) + self.free_width
        map_height = ceil(map_y_max) + self.free_width

        self.map = np.zeros((map_width, map_height), dtype=np.uint8)

        for polygon in copy_polygon_list:
            pts = polygon[:, :2].astype(int)
            pts = pts[..., ::-1]
            if fill:
                cv2.fillPoly(self.map, [pts], 255)
            else:
                cv2.polylines(self.map, [pts], True, 255)
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

    def generateLayoutFloor(self,
                            explore_map=None,
                            unit_size=0.01,
                            free_width=50,
                            dist_max=4,
                            render=False):
        self.updateMap(unit_size, free_width)

        if explore_map is None:
            boundary = getBoundary(self.map, 'box', dist_max, render)
        else:
            boundary = getBoundary(explore_map.map, 'polygon', dist_max,
                                   render)

        floor_point_list = []
        for pixel in boundary:
            if explore_map is None:
                point = self.getPointFromPixel(pixel[0], pixel[1])
            else:
                point = explore_map.getPointFromPixel(pixel[0], pixel[1])
            floor_point_list.append([point[0], point[1], 0.0])

        self.floor_array = np.array(floor_point_list)

        if render:
            renderPolygonAndFloor(self.polygon_list, self.floor_array)
        return True

    def generateLayoutMesh(self,
                           explore_map=None,
                           unit_size=0.01,
                           free_width=50,
                           wall_height=3,
                           dist_max=4,
                           skip_floor=False,
                           render=False):
        self.generateLayoutFloor(explore_map, unit_size, free_width, dist_max,
                                 render)

        layout_mesh = generateLayoutMesh(self.floor_array,
                                         wall_height,
                                         skip_floor=skip_floor)

        if render:
            window_name = '[LayoutMap::generateLayoutMesh]layout_mesh'
            if skip_floor:
                window_name += '_w/o_floor'
            drawGeometries([layout_mesh], window_name)
        return layout_mesh
