#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from random import randint

from scannet_sim_manage.Config.depth import INF, INF_POINT

from scannet_sim_manage.Data.bbox import BBox
from scannet_sim_manage.Data.point import Point

from scannet_sim_manage.Method.depth import getPointArray


class PointImage(object):
    def __init__(self, observations=None, agent_state=None):
        self.image = None
        self.depth = None
        self.point_array = None
        self.camera_point = None
        self.camera_face_to_point = None
        self.boundary_point_array = None
        self.bbox_2d_dict = {}
        self.label_dict_list = []

        if observations is not None and agent_state is not None:
            self.loadObservations(observations, agent_state)
        return

    def loadObservations(self, observations, agent_state, boundary_length=0.5):
        self.image = observations["color_sensor"][..., :3][..., ::-1]
        self.depth = observations["depth_sensor"]
        self.point_array, self.camera_point, self.boundary_point_array, self.camera_face_to_point = getPointArray(
            observations, agent_state, boundary_length)
        self.label_dict_list = [{} for _ in self.point_array]

        match_x_idx = np.where(self.point_array[:,
                                                0] == self.camera_point[0])[0]
        match_x_point_array = self.point_array[match_x_idx]

        x_match_y_idx = np.where(
            match_x_point_array[:, 1] == self.camera_point[1])[0]
        match_xy_idx = match_x_idx[x_match_y_idx]
        match_xy_point_array = self.point_array[match_xy_idx]

        xy_match_z_idx = np.where(
            match_xy_point_array[:, 2] == self.camera_point[2])[0]
        match_xyz_idx = match_xy_idx[xy_match_z_idx]

        self.point_array[match_xyz_idx] = INF_POINT

        for empty_idx in match_xyz_idx:
            self.addLabel(empty_idx, "empty")
        return True

    def getArrayIdx(self, pixel_idx):
        assert self.image is not None
        return pixel_idx[0] * self.image.shape[1] + pixel_idx[1]

    def getPixelIdx(self, array_idx):
        assert self.image is not None
        return [
            int(array_idx / self.image.shape[1]),
            array_idx % self.image.shape[1]
        ]

    def addLabel(self, array_idx, label, value=True):
        assert self.image is not None
        assert array_idx < len(self.point_array)
        assert "empty" not in self.label_dict_list[array_idx].keys()
        self.label_dict_list[array_idx][label] = value
        return True

    def addLabelMask(self, mask, label, value=True):
        assert self.image is not None
        assert self.image.shape[:2] == mask.shape

        mask_pixel_idx_array = np.dstack(np.where(mask == True))[0]
        for mask_pixel_idx in mask_pixel_idx_array:
            array_idx = self.getArrayIdx(mask_pixel_idx)
            if "empty" in self.label_dict_list[array_idx].keys():
                continue

            self.addLabel(array_idx, label, value)
        return True

    def getLabelBBox2D(self, label, value=True):
        x_min = INF
        x_max = -INF
        y_min = INF
        y_max = -INF

        for i, label_dict in enumerate(self.label_dict_list):
            if label not in label_dict.keys():
                continue
            if label_dict[label] != value:
                continue
            x, y = self.getPixelIdx(i)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        label_bbox = BBox(Point(x_min, y_min, 0), Point(x_max, y_max, 0))
        return label_bbox

    def getValidLabelValueList(self):
        valid_label_value_list = []
        for label_dict in self.label_dict_list:
            if "empty" in label_dict.keys():
                continue
            for label, value in label_dict.items():
                if [label, value] in valid_label_value_list or \
                        label == "background" or \
                        value not in ["object", True]:
                    continue

                valid_label_value_list.append([label, value])
        return valid_label_value_list

    def updateAllLabelBBox2D(self):
        label_value_list = self.getValidLabelValueList()

        for label, value in label_value_list:
            self.bbox_2d_dict[label] = self.getLabelBBox2D(label, value)
        return True

    def getLabelImage(self, image, label, value=True):
        label_image = np.ones(image.shape, dtype=image.dtype)

        if image.dtype == np.uint8:
            label_image *= 255

        for i, label_dict in enumerate(self.label_dict_list):
            if label not in label_dict.keys():
                continue
            if label_dict[label] != value:
                continue
            pixel_idx = self.getPixelIdx(i)
            if image.dtype == np.uint8:
                label_image[pixel_idx[0], pixel_idx[1], :] = \
                    image[pixel_idx[0],  pixel_idx[1], :]
            else:
                label_image[pixel_idx[0], pixel_idx[1]] = \
                    image[pixel_idx[0],  pixel_idx[1]]
        return label_image

    def getLabelRGB(self, label, value=True):
        return self.getLabelImage(self.image, label, value)

    def getLabelDepth(self, label, value=True):
        return self.getLabelImage(self.depth, label, value)

    def getAllLabelImage(self, with_color=False):
        if with_color:
            all_label_image = np.ones(self.image.shape, dtype=np.uint8) * 255
        else:
            all_label_image = np.zeros(self.image.shape[:2], dtype=int)

        if with_color:
            background_color = [0, 0, 0]
            background_color = None
        else:
            background_color = -1

        label_list = self.getValidLabelValueList()
        if with_color:
            color_list = [[randint(0, 255),
                           randint(0, 255),
                           randint(0, 255)] for _ in label_list]
        else:
            color_list = [i for i in range(len(label_list))]

        for i, label_dict in enumerate(self.label_dict_list):
            if "empty" in label_dict.keys():
                continue

            if "background" in label_dict.keys():
                pixel_idx = self.getPixelIdx(i)
                if with_color:
                    if background_color is None:
                        #  all_label_image[pixel_idx[0], pixel_idx[1]] = \
                        #  self.image[pixel_idx[0], pixel_idx[1]]

                        source_color = self.image[pixel_idx[0], pixel_idx[1]]
                        gray_value = 0.3 * source_color[
                            2] + 0.59 * source_color[1] + 0.11 * source_color[0]
                        gray_color = [gray_value, gray_value, gray_value]
                        all_label_image[pixel_idx[0],
                                        pixel_idx[1]] = gray_color
                    else:
                        all_label_image[pixel_idx[0], pixel_idx[1], :] = \
                            background_color
                else:
                    all_label_image[pixel_idx[0], pixel_idx[1]] = \
                        background_color
                continue

            for j, [label, value] in enumerate(label_list):
                if label not in label_dict.keys():
                    continue
                if label_dict[label] != value:
                    continue
                if value not in ["object", True]:
                    continue
                pixel_idx = self.getPixelIdx(i)
                if with_color:
                    all_label_image[pixel_idx[0], pixel_idx[1], :] = \
                        color_list[j]
                else:
                    all_label_image[pixel_idx[0], pixel_idx[1]] = \
                        color_list[j]
                break
        return all_label_image

    def getAllLabelMask(self):
        return self.getAllLabelImage()

    def getAllLabelRender(self):
        return self.getAllLabelImage(True)
