#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import struct

import cv2
import imageio
import numpy as np
import png
from tqdm import tqdm

from scannet_dataset_manage.Config.sensor_data import (COMPRESSION_TYPE_COLOR,
                                                       COMPRESSION_TYPE_DEPTH)
from scannet_dataset_manage.Data.rgbd_frame import RGBDFrame
from scannet_dataset_manage.Data.sensor_data import SensorData


class SensReader(SensorData):
    def __init__(self, sens_file_path=None, header_only=True):
        super().__init__()
        self.version = 4

        self.num_frames = None

        if sens_file_path is not None:
            self.loadSensFile(sens_file_path, header_only)
        return

    def reset(self):
        super().reset()

        self.num_frames = None
        return True

    def loadSensFile(self, sens_file_path, header_only=True):
        self.reset()

        assert os.path.exists(sens_file_path)

        with open(sens_file_path, 'rb') as f:
            version = struct.unpack('I', f.read(4))[0]
            if version != self.version:
                print('[ERROR][SensReader::loadSensFile]')
                print('\t version not matched! need to be ' +
                      str(self.version) + '!')
                return False

            strlen = struct.unpack('Q', f.read(8))[0]

            self.sensor_name = bytes('', encoding='utf-8').join(
                struct.unpack('c' * strlen, f.read(strlen)))

            self.intrinsic_color = np.asarray(struct.unpack(
                'f' * 16, f.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.extrinsic_color = np.asarray(struct.unpack(
                'f' * 16, f.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.intrinsic_depth = np.asarray(struct.unpack(
                'f' * 16, f.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)
            self.extrinsic_depth = np.asarray(struct.unpack(
                'f' * 16, f.read(16 * 4)),
                                              dtype=np.float32).reshape(4, 4)

            self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack(
                'i', f.read(4))[0]]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack(
                'i', f.read(4))[0]]

            self.color_width = struct.unpack('I', f.read(4))[0]
            self.color_height = struct.unpack('I', f.read(4))[0]
            self.depth_width = struct.unpack('I', f.read(4))[0]
            self.depth_height = struct.unpack('I', f.read(4))[0]
            self.depth_shift = struct.unpack('f', f.read(4))[0]

            self.num_frames = struct.unpack('Q', f.read(8))[0]

            if header_only:
                return True

            print('[INFO][SensReader::loadSensFile]')
            print('\t start load frames...')
            for _ in tqdm(range(self.num_frames)):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)
        return True

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        print('exporting',
              len(self.frames) // frame_skip, ' depth frames to', output_path)
        for f in tqdm(range(0, len(self.frames), frame_skip)):
            depth_data = self.frames[f].decompress_depth(
                self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width)
            if image_size is not None:
                depth = cv2.resize(depth, (image_size[1], image_size[0]),
                                   interpolation=cv2.INTER_NEAREST)
            #imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)
            with open(os.path.join(output_path,
                                   str(f) + '.png'),
                      'wb') as f:  # write 16-bit
                writer = png.Writer(width=depth.shape[1],
                                    height=depth.shape[0],
                                    bitdepth=16)
                depth = depth.reshape(-1, depth.shape[1]).tolist()
                writer.write(f, depth)
        return True

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        print('exporting',
              len(self.frames) // frame_skip, 'color frames to', output_path)
        for f in tqdm(range(0, len(self.frames), frame_skip)):
            color = self.frames[f].decompress_color(
                self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(color, (image_size[1], image_size[0]),
                                   interpolation=cv2.INTER_NEAREST)
            imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)
        return True

    def save_mat_to_file(self, matrix, filename):
        with open(filename, 'w') as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt='%f')
        return True

    def export_poses(self, output_path, frame_skip=1):
        os.makedirs(output_path, exist_ok=True)
        print('exporting',
              len(self.frames) // frame_skip, 'camera poses to', output_path)
        for f in tqdm(range(0, len(self.frames), frame_skip)):
            self.save_mat_to_file(self.frames[f].camera_to_world,
                                  os.path.join(output_path,
                                               str(f) + '.txt'))
        return True

    def export_intrinsics(self, output_path):
        os.makedirs(output_path, exist_ok=True)
        print('exporting camera intrinsics to', output_path)
        self.save_mat_to_file(self.intrinsic_color,
                              os.path.join(output_path, 'intrinsic_color.txt'))
        self.save_mat_to_file(self.extrinsic_color,
                              os.path.join(output_path, 'extrinsic_color.txt'))
        self.save_mat_to_file(self.intrinsic_depth,
                              os.path.join(output_path, 'intrinsic_depth.txt'))
        self.save_mat_to_file(self.extrinsic_depth,
                              os.path.join(output_path, 'extrinsic_depth.txt'))
        return True
