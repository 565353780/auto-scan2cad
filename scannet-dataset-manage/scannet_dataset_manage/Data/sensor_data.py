#!/usr/bin/env python
# -*- coding: utf-8 -*-


class SensorData:
    def __init__(self):
        self.sensor_name = None

        self.intrinsic_color = None
        self.extrinsic_color = None
        self.intrinsic_depth = None
        self.extrinsic_depth = None

        self.color_compression_type = None
        self.depth_compression_type = None
        self.depth_width = None
        self.depth_height = None
        self.depth_shift = None

        self.frames = []
        return

    def reset(self):
        self.sensor_name = None

        self.intrinsic_color = None
        self.extrinsic_color = None
        self.intrinsic_depth = None
        self.extrinsic_depth = None

        self.color_compression_type = None
        self.depth_compression_type = None
        self.depth_width = None
        self.depth_height = None
        self.depth_shift = None

        self.frames = []
        return True
