#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import open3d as o3d
from getch import getch
from tqdm import tqdm
from math import cos, sin, pi

from auto_cad_recon.Method.io import loadDataList
from auto_cad_recon.Method.render import getRenderGeometries


class Renderer(object):

    def __init__(self):
        self.data_list_list = []

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()

        self.euler_angle = [0.0, 0.0, 0.0]
        self.zoom = 1
        return

    def reset(self):
        self.data_list_list = []
        return True

    def closeVis(self):
        self.vis.destroy_window()
        return True

    def loadResult(self, data_list_json_folder_path, print_progress=False):
        assert os.path.exists(data_list_json_folder_path)

        self.reset()

        json_file_name_list = os.listdir(data_list_json_folder_path)

        json_file_idx_list = []
        for json_file_name in json_file_name_list:
            if json_file_name[-5:] != ".json":
                continue
            json_file_idx_list.append(int(json_file_name[:-5]))
        json_file_idx_list.sort()

        if len(json_file_idx_list) == 0:
            print("[ERROR][Renderer::loadResult]")
            print("\t json_file not exist!")
            print("\t folder:", data_list_json_folder_path)
            return False

        for_data = json_file_idx_list
        if print_progress:
            print("[INFO][Renderer::loadResult]")
            print("\t start load result json files...")
            for_data = tqdm(for_data)

        for json_file_idx in for_data:
            json_file_path = data_list_json_folder_path + str(
                json_file_idx) + ".json"
            self.data_list_list.append(loadDataList(json_file_path))
        return True

    def getRotationMatrixFromEulerAngle(self, euler_angle):
        R_x = np.array([[1, 0, 0],
                        [0, cos(euler_angle[0]), -sin(euler_angle[0])],
                        [0, sin(euler_angle[0]),
                         cos(euler_angle[0])]])

        R_y = np.array([[cos(euler_angle[1]), 0,
                         sin(euler_angle[1])], [0, 1, 0],
                        [-sin(euler_angle[1]), 0,
                         cos(euler_angle[1])]])

        R_z = np.array([[cos(euler_angle[2]), -sin(euler_angle[2]), 0],
                        [sin(euler_angle[2]),
                         cos(euler_angle[2]), 0], [0, 0, 1]])

        rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
        return rotation_matrix

    def getRotateDirection(self, direction_vector, euler_angle):
        np_direction_vector = np.array(direction_vector)
        direction_vector_norm = np.linalg.norm(np_direction_vector)
        if direction_vector_norm == 0:
            print("[ERROR][Renderer::getRotateDirection]")
            print("\t direction_vector_norm is 0!")
            return None

        np_unit_direction_vector = np_direction_vector / direction_vector_norm

        rotation_matrix = self.getRotationMatrixFromEulerAngle(euler_angle)

        rotate_direction = np.dot(rotation_matrix, np_unit_direction_vector)
        return rotate_direction.tolist()

    def rotateVis(self, delta_rotate_angle, delta_zoom):
        for i in range(3):
            self.euler_angle[i] += delta_rotate_angle[i] * pi / 180.0
        self.zoom *= delta_zoom

        ctr = self.vis.get_view_control()

        front_direction = self.getRotateDirection([1, 0, 0], self.euler_angle)
        ctr.set_front(front_direction)

        up_direction = self.getRotateDirection([0, 0, 1], self.euler_angle)
        ctr.set_up(up_direction)

        ctr.set_zoom(self.zoom)

        self.vis.poll_events()
        self.vis.update_renderer()
        return True

    def rotateX(self, delta_angle):
        return self.rotateVis([delta_angle, 0.0, 0.0], 1.0)

    def rotateY(self, delta_angle):
        return self.rotateVis([0.0, delta_angle, 0.0], 1.0)

    def rotateZ(self, delta_angle):
        return self.rotateVis([0.0, 0.0, delta_angle], 1.0)

    def scaleZoom(self, delta_zoom):
        return self.rotateVis([0.0, 0.0, 0.0], delta_zoom)

    def renderResultWithIdx(self, render_idx):
        assert render_idx < len(self.data_list_list)

        self.vis.clear_geometries()

        mesh_list = getRenderGeometries(self.data_list_list[render_idx])
        for mesh in mesh_list:
            self.vis.add_geometry(mesh)

        self.rotateVis([0.0, 0.0, 0.0], 1.0)
        return True

    def renderResult(self, data_list_json_folder_path, print_progress=False):
        if not self.loadResult(data_list_json_folder_path, print_progress):
            print("[ERROR][Renderer::renderResult]")
            print("\t loadResult failed!")
            return False

        render_idx = 0

        self.renderResultWithIdx(render_idx)

        while True:
            key = getch()
            if key == "w":
                render_idx = 0
                self.renderResultWithIdx(render_idx)
                continue
            if key == "r":
                render_idx = len(self.data_list_list) - 1
                self.renderResultWithIdx(render_idx)
                continue
            if key == "s":
                render_idx = max(render_idx - 1, 0)
                self.renderResultWithIdx(render_idx)
                continue
            if key == "f":
                render_idx = min(render_idx + 1, len(self.data_list_list) - 1)
                self.renderResultWithIdx(render_idx)
                continue
            if key == "j":
                self.rotateZ(-5.0)
                continue
            if key == "l":
                self.rotateZ(5.0)
                continue
            if key == "i":
                self.rotateY(-5.0)
                continue
            if key == "k":
                self.rotateY(5.0)
                continue
            if key == "u":
                self.scaleZoom(1.1)
                continue
            if key == "o":
                self.scaleZoom(0.9)
                continue
            if key == "q":
                self.closeVis()
                break
        return True
