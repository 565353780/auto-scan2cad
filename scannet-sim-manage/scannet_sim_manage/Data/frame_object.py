#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from copy import deepcopy

from scannet_sim_manage.Config.depth import INF_POINT


class FrameObject():

    def __init__(self, point_image=None, label=None, value=True):
        self.image = None
        self.depth = None
        self.point_array = []
        self.camera_point = None
        self.bbox_2d = None
        self.label_dict_list = []

        if point_image is not None and label is not None:
            self.generateData(point_image, label, value)
        return

    def generateData(self, point_image, label, value=True):
        self.image = point_image.getLabelRGB(label, value)

        self.depth = point_image.getLabelDepth(label, value)

        self.camera_point = point_image.camera_point

        assert label in point_image.bbox_2d_dict.keys()
        self.bbox_2d = point_image.bbox_2d_dict[label]

        for i, label_dict in enumerate(point_image.label_dict_list):
            if label not in label_dict.keys():
                self.point_array.append(INF_POINT)
                self.label_dict_list.append({"empty": True})
                continue
            if label_dict[label] != value:
                self.point_array.append(INF_POINT)
                self.label_dict_list.append({"empty": True})
                continue
            self.point_array.append(point_image.point_array[i])
            self.label_dict_list.append(point_image.label_dict_list[i])

        self.point_array = np.array(self.point_array, dtype=float)
        return True

    def getBBoxImage(self, width, height, free_width):
        bbox_image = np.ones((width, height, 3), dtype=np.uint8) * 255

        x1 = self.bbox_2d.min_point.x
        y1 = self.bbox_2d.min_point.y
        x2 = self.bbox_2d.max_point.x
        y2 = self.bbox_2d.max_point.y

        image_copy = deepcopy(self.image[x1:x2, y1:y2])

        x_mean = (x1 + x2) / 2.0
        y_mean = (y1 + y2) / 2.0
        x_diff = x2 - x1
        y_diff = y2 - y1

        x_diff = max(x_diff, 1)
        y_diff = max(y_diff, 1)

        scale = min(1.0 * (width - 2.0 * free_width) / y_diff,
                    1.0 * (height - 2.0 * free_width) / x_diff)

        scaled_image_copy = cv2.resize(image_copy,
                                       None,
                                       fx=scale,
                                       fy=scale,
                                       interpolation=cv2.INTER_CUBIC)

        y_start = int(height / 2.0 + scale * (x1 - x_mean))
        x_start = int(width / 2.0 + scale * (y1 - y_mean))
        x_end = x_start + scaled_image_copy.shape[1]
        y_end = y_start + scaled_image_copy.shape[0]

        bbox_image[y_start:y_end, x_start:x_end] = scaled_image_copy
        return bbox_image
