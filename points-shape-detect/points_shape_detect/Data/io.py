#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import open3d as o3d


class IO:

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)

    @classmethod
    def _read_pcd(cls, file_path):
        pc = o3d.io.read_point_cloud(file_path)
        points = np.array(pc.points)
        return points

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]

    @classmethod
    def _read_mesh(cls, file_path, trans_matrix=None):
        mesh = o3d.io.read_triangle_mesh(file_path)
        pcd = mesh.sample_points_uniformly(8192)
        if trans_matrix is not None:
            pcd.transform(trans_matrix)
        points = np.array(pcd.points)
        return points

    @classmethod
    def get(cls, file_path, trans_matrix=None):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.ply', '.obj']:
            return cls._read_mesh(file_path, trans_matrix)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)
