#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from copy import deepcopy


class PolylineRenderer(object):

    def __init__(self,
                 width=1920,
                 height=1080,
                 free_width=50,
                 render_width=2560,
                 render_height=1440,
                 debug=False):
        self.render_mode_list = ['source', 'cluster']

        self.width = width
        self.height = height
        self.free_width = free_width
        self.render_width = render_width
        self.render_height = render_height

        self.x_min = float('inf')
        self.y_min = float('inf')
        self.x_max = -float('inf')
        self.y_max = -float('inf')

        self.origin_translate = None
        self.scale = None
        self.post_translate = np.array([self.width / 2.0, self.height / 2.0],
                                       dtype=float)

        self.image = None
        self.resetImage()

        self.image_list = []
        self.text_color = [0, 0, 255]
        self.text_line_width = 1

        self.cluster_idx_list = None

        self.debug = debug
        return

    def resetImage(self):
        self.image = np.zeros([self.height, self.width, 3], dtype=np.uint8)
        return True

    def resetTransform(self):
        self.x_min = float('inf')
        self.y_min = float('inf')
        self.x_max = -float('inf')
        self.y_max = -float('inf')

        self.origin_translate = None
        self.scale = None

        self.resetImage()
        return True

    def reset(self):
        self.resetTransform()

        self.cluster_idx_list = None
        return True

    def addPoint(self, point):
        self.x_min = min(self.x_min, point[0])
        self.y_min = min(self.y_min, point[1])
        self.x_max = max(self.x_max, point[0])
        self.y_max = max(self.y_max, point[1])
        return True

    def updateViewBox(self, polyline):
        for point in polyline:
            self.addPoint(point)
        return True

    def updateTransform(self, polyline):
        self.resetTransform()

        self.updateViewBox(polyline)

        self.origin_translate = np.array([
            -(self.x_min + self.x_max) / 2.0, -(self.y_min + self.y_max) / 2.0
        ],
                                         dtype=float)

        self.scale = min(
            (self.width - 2.0 * self.free_width) / (self.x_max - self.x_min),
            (self.height - 2.0 * self.free_width) / (self.y_max - self.y_min))
        return True

    def getPointInImage(self, point_in_world):
        point_in_image = np.array(deepcopy(point_in_world), dtype=float)
        point_in_image += self.origin_translate
        point_in_image *= self.scale
        point_in_image += self.post_translate
        point_in_image = point_in_image.astype(int)
        return point_in_image

    def getPointInWorld(self, point_in_image):
        point_in_world = np.array(deepcopy(point_in_image), dtype=float)
        point_in_world -= self.post_translate
        point_in_world /= self.scale
        point_in_world -= self.origin_translate
        return point_in_world

    def renderLine(self,
                   start,
                   end,
                   color,
                   line_width,
                   put_text=None,
                   text_color=[0, 0, 255],
                   text_size=1,
                   text_line_width=1):
        start_in_image = self.getPointInImage(start)
        end_in_image = self.getPointInImage(end)

        cv2.line(self.image, start_in_image, end_in_image, color, line_width)

        if put_text is not None:
            cv2.putText(self.image, put_text,
                        ((start_in_image + end_in_image) / 2.0).astype(int),
                        cv2.FONT_HERSHEY_COMPLEX, text_size, text_color,
                        text_line_width)
        return True

    def updateImageBySource(self,
                            polyline,
                            line_width=1,
                            save_into_list=False,
                            text_color=[0, 0, 255],
                            text_size=1,
                            text_line_width=1):
        line_color = [255, 255, 255]

        point_num = len(polyline)

        for i in range(point_num):
            next_point_idx = (i + 1) % point_num

            if self.debug:
                put_text = str(i)
            else:
                put_text = None

            self.renderLine(polyline[i], polyline[next_point_idx], line_color,
                            line_width, put_text, text_color, text_size,
                            text_line_width)

        if save_into_list:
            self.image_list.append(deepcopy(self.image))
        return True

    def updateImageByCluster(self,
                             polyline,
                             line_width=1,
                             save_into_list=False,
                             text_color=[0, 0, 255],
                             text_size=1,
                             text_line_width=1):
        unit_cluster_idx_list = sorted(list(set(self.cluster_idx_list)))
        cluster_color_dict = {}
        for unit_cluster_idx in unit_cluster_idx_list:
            cluster_color_dict[str(unit_cluster_idx)] = np.array(
                [
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                ],
                dtype=np.uint8).tolist()

        point_num = len(polyline)

        for i in range(point_num):
            next_point_idx = (i + 1) % point_num
            cluster_idx = self.cluster_idx_list[i]

            if self.debug:
                put_text = str(cluster_idx)
            else:
                put_text = None

            self.renderLine(polyline[i], polyline[next_point_idx],
                            cluster_color_dict[str(cluster_idx)], line_width,
                            put_text, text_color, text_size, text_line_width)

        if save_into_list:
            self.image_list.append(deepcopy(self.image))
        return True

    def updateImage(self,
                    polyline,
                    render_mode='source',
                    line_width=1,
                    text_color=[0, 0, 255],
                    text_size=1,
                    text_line_width=1,
                    save_into_list=False):
        self.updateTransform(polyline)

        if '+' in render_mode:
            sub_render_mode_list = render_mode.split('+')
            for sub_render_mode in sub_render_mode_list:
                self.updateImage(polyline, sub_render_mode, line_width,
                                 text_color, text_size, text_line_width, True)
            return True

        assert render_mode in self.render_mode_list

        if render_mode == 'source':
            return self.updateImageBySource(polyline, line_width,
                                            save_into_list, text_color,
                                            text_size, text_line_width)
        elif render_mode == 'cluster':
            return self.updateImageByCluster(polyline, line_width,
                                             save_into_list, text_color,
                                             text_size, text_line_width)
        return True

    def render(self,
               polyline,
               render_mode='source',
               line_width=1,
               text_color=[0, 0, 255],
               text_size=1,
               text_line_width=1,
               cluster_idx_list=None):
        self.cluster_idx_list = cluster_idx_list

        self.image_list = []

        self.updateImage(polyline, render_mode, line_width, text_color,
                         text_size, text_line_width)
        return True

    def getRenderImage(self):
        if len(self.image_list) == 0:
            return self.image

        render_image = np.hstack(self.image_list)
        return render_image

    def getRenderImageList(self):
        if len(self.image_list) == 0:
            return [self.image]

        return self.image_list

    def show(self, wait_key=0, window_name='[PolylineRenderer][image]'):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.render_width, self.render_height)

        render_image = self.getRenderImage()

        cv2.imshow(window_name, render_image)
        cv2.waitKey(wait_key)
        return True
