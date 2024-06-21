#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import open3d as o3d
from math import cos, sin, pi

from mesh_manage.Method.path import createFileFolder


class Renderer(object):

    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.render_center = None
        self.euler_angle = [0, 0, 0]
        return

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

    def rotateVis(self, delta_rotate_angle):
        self.euler_angle[0] = 0
        self.euler_angle[1] = -10 * pi / 180.0
        self.euler_angle[2] += delta_rotate_angle * pi / 180.0

        # FIXME: only for demo
        self.euler_angle[0] = -90 * pi / 180.0
        self.euler_angle[1] = 0 * pi / 180.0
        self.euler_angle[2] = 50 * pi / 180.0

        ctr = self.vis.get_view_control()

        front_direction = self.getRotateDirection([1, 0, 0], self.euler_angle)
        ctr.set_front(front_direction)

        up_direction = self.getRotateDirection([0, 0, 1], self.euler_angle)
        ctr.set_up(up_direction)

        ctr.set_lookat(self.render_center)
        ctr.set_zoom(0.12)
        return True

    def renderMesh(self, mesh_file_path):
        delta_rotate_angle = 0.5

        assert os.path.exists(mesh_file_path)

        mesh = o3d.io.read_triangle_mesh(mesh_file_path)

        self.render_center = mesh.get_axis_aligned_bounding_box().get_center()
        self.render_center[0] += 18
        self.render_center[2] -= 2

        bbox_points = np.array(mesh.get_axis_aligned_bounding_box().get_box_points()).tolist()

        colors = np.array(mesh.vertex_colors)
        gray_points_idx = np.where(colors[:, 0] > 130.0 / 255.0)[0]
        mesh.remove_vertices_by_index(gray_points_idx)

        points = np.array(mesh.vertices).tolist()
        for bbox_point in bbox_points:
            points.append(bbox_point)

        colors = np.array(mesh.vertex_colors).tolist()
        for bbox_point in bbox_points:
            colors.append([1.0, 1.0, 1.0])

        mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))

        mesh.compute_vertex_normals()

        self.vis.create_window(window_name="Renderer")
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([1, 1, 1])
        render_option.point_size = 1

        self.vis.add_geometry(mesh)
        while True:
            self.rotateVis(delta_rotate_angle)
            #  self.vis.update_geometry()
            self.vis.poll_events()
            self.vis.update_renderer()

            if ord('q') == cv2.waitKey(1):
                break
        self.vis.destroy_window()
        return True

    def saveRenderMeshImage(self, mesh_file_path, save_image_file_path):
        assert os.path.exists(mesh_file_path)

        mesh = o3d.io.read_triangle_mesh(mesh_file_path)

        self.render_center = mesh.get_axis_aligned_bounding_box().get_center()
        self.render_center[0] += 18
        self.render_center[2] -= 2

        bbox_points = np.array(mesh.get_axis_aligned_bounding_box().get_box_points()).tolist()

        colors = np.array(mesh.vertex_colors)
        gray_points_idx = np.where(colors[:, 0] > 130.0 / 255.0)[0]
        mesh.remove_vertices_by_index(gray_points_idx)

        points = np.array(mesh.vertices).tolist()
        for bbox_point in bbox_points:
            points.append(bbox_point)

        colors = np.array(mesh.vertex_colors).tolist()
        for bbox_point in bbox_points:
            colors.append([1.0, 1.0, 1.0])

        mesh.vertices = o3d.utility.Vector3dVector(np.array(points))
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(colors))

        mesh.compute_vertex_normals()

        self.vis.create_window(window_name="Renderer")
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([1, 1, 1])
        render_option.point_size = 1

        self.vis.add_geometry(mesh)

        self.rotateVis(0.5)
        #  self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

        open3d_image = np.asarray(
            self.vis.capture_screen_float_buffer()) * 255.0
        cv_image = cv2.cvtColor(open3d_image,
                                cv2.COLOR_RGB2BGR).astype(np.uint8)

        createFileFolder(save_image_file_path)

        cv2.imwrite(save_image_file_path, cv_image)

        self.vis.destroy_window()
        return True

    def saveRenderMesh(self, mesh_file_path, output_video_file_path):
        fps = 30
        video_width = 1920
        video_height = 1080
        delta_rotate_angle = 0.5

        assert os.path.exists(mesh_file_path)

        mesh = o3d.io.read_triangle_mesh(mesh_file_path)

        self.render_center = mesh.get_axis_aligned_bounding_box().get_center()

        self.vis.create_window(window_name="Renderer")
        render_option = self.vis.get_render_option()
        render_option.background_color = np.array([1, 1, 1])
        render_option.point_size = 1

        self.vis.add_geometry(mesh)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(output_video_file_path, fourcc, fps,
                              (video_width, video_height))
        for i in range(int(360 / delta_rotate_angle)):
            self.rotateVis(0.5)
            #  self.vis.update_geometry()
            self.vis.poll_events()
            self.vis.update_renderer()

            open3d_image = np.asarray(
                self.vis.capture_screen_float_buffer()) * 255.0
            cv_image = cv2.cvtColor(open3d_image,
                                    cv2.COLOR_RGB2BGR).astype(np.uint8)

            out.write(cv_image)

        self.vis.destroy_window()
        out.release()
        return True
