#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import open3d as o3d
from time import time
from getch import getch
from copy import deepcopy

from auto_cad_recon.Method.scene_render import getMergedScenePCD
from auto_cad_recon.Method.render import renderMergedScene
from auto_cad_recon.Module.dataset_manager import DatasetManager
from auto_cad_recon.Module.dataset_render_manager import DatasetRenderManager


class SimpleSceneExplorer(object):
    def __init__(
        self,
        scannet_dataset_folder_path,
        scannet_glb_dataset_folder_path,
        scannet_object_dataset_folder_path,
        scannet_bbox_dataset_folder_path,
        scan2cad_dataset_folder_path,
        scan2cad_object_model_map_dataset_folder_path,
        shapenet_dataset_folder_path,
        shapenet_udf_dataset_folder_path,
    ):
        self.dataset_manager = DatasetManager(
            scannet_dataset_folder_path,
            scannet_object_dataset_folder_path,
            scannet_bbox_dataset_folder_path,
            scan2cad_dataset_folder_path,
            scan2cad_object_model_map_dataset_folder_path,
            shapenet_dataset_folder_path,
            shapenet_udf_dataset_folder_path,
        )
        self.dataset_render_manager = DatasetRenderManager(
            scannet_glb_dataset_folder_path,
            scannet_object_dataset_folder_path,
            scannet_bbox_dataset_folder_path,
        )

        self.scannet_scene_name = None
        return

    def generateFullDataset(self):
        return self.dataset_manager.generateFullDataset()

    def getScanNetSceneNameList(self):
        return self.dataset_manager.getScanNetSceneNameList()

    def isSceneValid(self, scannet_scene_name):
        return self.dataset_manager.isSceneValid(scannet_scene_name)

    def getScanNetObjectFileNameList(self, scannet_scene_name):
        return self.dataset_manager.getScanNetObjectFileNameList(scannet_scene_name)

    def isObjectValid(self, scannet_scene_name, scannet_object_file_name):
        return self.dataset_manager.isObjectValid(
            scannet_scene_name, scannet_object_file_name
        )

    def getShapeNetModelDict(self, scannet_scene_name, scannet_object_file_name):
        return self.dataset_manager.getShapeNetModelDict(
            scannet_scene_name, scannet_object_file_name
        )

    def getShapeNetModelTensorDict(self, scannet_scene_name, scannet_object_file_name):
        return self.dataset_manager.getShapeNetModelTensorDict(
            scannet_scene_name, scannet_object_file_name
        )

    def loadScene(self, scannet_scene_name, print_progress=False):
        self.scannet_scene_name = scannet_scene_name
        valid_object_file_name_list = self.getScanNetObjectFileNameList(
            self.scannet_scene_name
        )
        self.dataset_render_manager.loadScene(
            self.scannet_scene_name, print_progress, valid_object_file_name_list
        )
        return True

    def setControlMode(self, control_mode):
        return self.dataset_render_manager.setControlMode(control_mode)

    def setAgentPose(self, xyz_list, urf_list):
        return self.dataset_render_manager.setAgentPose(xyz_list, urf_list)

    def getSceneObjectLabelList(self):
        return self.dataset_render_manager.getSceneObjectLabelList()

    def getFrameObjectDict(self, object_label):
        return self.dataset_render_manager.getFrameObjectDict(object_label)

    def saveAllSceneObjects(
        self,
        scene_objects_save_folder_path,
        bbox_image_width,
        bbox_image_height,
        bbox_image_free_width,
    ):
        return self.dataset_render_manager.saveAllSceneObjects(
            scene_objects_save_folder_path,
            bbox_image_width,
            bbox_image_height,
            bbox_image_free_width,
        )

    def saveCurrentObservation(self, render=False):
        point_image_list = (
            self.dataset_render_manager.scannet_sim_loader.point_image_list
        )
        explore_map = self.dataset_render_manager.scannet_sim_loader.layout_map_builder.explore_map

        root_save_folder_path = (
            "./output/simple_scene_explorer/" + self.scannet_scene_name
        )

        observation_save_folder_path = root_save_folder_path + "/observation/"
        os.makedirs(observation_save_folder_path, exist_ok=True)
        explore_map_save_folder_path = root_save_folder_path + "/explore_map/"
        os.makedirs(explore_map_save_folder_path, exist_ok=True)
        camera_position_save_folder_path = root_save_folder_path + "/camera_position/"
        os.makedirs(camera_position_save_folder_path, exist_ok=True)
        rgbd_save_folder_path = root_save_folder_path + "/rgbd/"
        os.makedirs(rgbd_save_folder_path, exist_ok=True)

        current_idx_str = str(len(point_image_list))

        scene_pcd = getMergedScenePCD([point_image_list[-1]], estimate_normals=True)

        o3d.io.write_point_cloud(
            observation_save_folder_path + current_idx_str + ".ply",
            scene_pcd,
            write_ascii=True,
        )

        camera_point = deepcopy(point_image_list[-1].camera_point).reshape(3)
        camera_face_to_point = deepcopy(
            point_image_list[-1].camera_face_to_point
        ).reshape(3)
        camera_point_in_image = explore_map.getPixelFromPoint(camera_point)
        camera_face_to_point_in_image = explore_map.getPixelFromPoint(
            camera_face_to_point
        )

        with open(
            camera_position_save_folder_path + current_idx_str + ".txt", "w"
        ) as f:
            f.write("camera_point:")
            f.write(
                str(camera_point[0])
                + ","
                + str(camera_point[1])
                + ","
                + str(camera_point[2])
                + "\n"
            )
            f.write("camera_face_to_point:")
            f.write(
                str(camera_face_to_point[0])
                + ","
                + str(camera_face_to_point[1])
                + ","
                + str(camera_face_to_point[2])
                + "\n"
            )
            f.write("camera_point_in_image:")
            f.write(
                str(camera_point_in_image[0])
                + ","
                + str(camera_point_in_image[1])
                + "\n"
            )
            f.write("camera_face_to_point_in_image:")
            f.write(
                str(camera_face_to_point_in_image[0])
                + ","
                + str(camera_face_to_point_in_image[1])
                + "\n"
            )

        explore_image = deepcopy(explore_map.map)
        cv2.imwrite(
            explore_map_save_folder_path + current_idx_str + ".png", explore_image
        )

        rgb = deepcopy(point_image_list[-1].image)
        cv2.imwrite(rgbd_save_folder_path + current_idx_str + "_rgb.png", rgb)
        depth = deepcopy(point_image_list[-1].depth)
        np.save(rgbd_save_folder_path + current_idx_str + "_depth.npy", depth)

        if render:
            cv2.arrowedLine(
                explore_image,
                camera_point_in_image[[1, 0]],
                camera_face_to_point_in_image[[1, 0]],
                (0, 0, 255),
                2,
                0,
                0,
                0.2,
            )
            cv2.imshow(
                "[SimpleSceneExplorer::renderMergedScene]explore_image", explore_image
            )
        return True

    def saveMergedScene(self, render=False):
        point_image_list = (
            self.dataset_render_manager.scannet_sim_loader.point_image_list
        )

        root_save_folder_path = (
            "./output/simple_scene_explorer/" + self.scannet_scene_name
        )

        scene_save_folder_path = root_save_folder_path + "/scene/"
        os.makedirs(scene_save_folder_path, exist_ok=True)

        current_idx_str = str(len(point_image_list))

        merged_scene_pcd = getMergedScenePCD(point_image_list, estimate_normals=True)

        o3d.io.write_point_cloud(
            scene_save_folder_path + current_idx_str + ".ply",
            merged_scene_pcd,
            write_ascii=True,
        )

        if render:
            renderMergedScene(point_image_list, estimate_normals=True)
        return True

    def startKeyBoardControlRender(self, wait_val, print_progress=False):
        #  self.dataset_render_manager.scannet_sim_loader.sim_manager.resetAgentPose()
        self.dataset_render_manager.scannet_sim_loader.sim_manager.cv_renderer.init()

        while True:
            if not self.dataset_render_manager.scannet_sim_loader.sim_manager.cv_renderer.renderFrame(
                self.dataset_render_manager.scannet_sim_loader.sim_manager.sim_loader.observations
            ):
                break
            self.dataset_render_manager.scannet_sim_loader.sim_manager.cv_renderer.waitKey(
                wait_val
            )

            agent_state = self.dataset_render_manager.scannet_sim_loader.sim_manager.sim_loader.getAgentState()
            print(
                "agent_state: position",
                agent_state.position,
                "rotation",
                agent_state.rotation,
            )

            input_key = getch()
            if input_key == "v":
                start_time = time()
                self.dataset_render_manager.scannet_sim_loader.getObjectInView(
                    print_progress
                )
                spend_time_ms = 1000 * (time() - start_time)
                print("fusion time spend:", spend_time_ms, "ms")
                start_time = time()
                self.saveCurrentObservation()
                spend_time_ms = 1000 * (time() - start_time)
                print("saveCurrentObservation time spend:", spend_time_ms, "ms")
                continue
            if input_key == "c":
                start_time = time()
                self.saveMergedScene()
                spend_time_ms = 1000 * (time() - start_time)
                print("saveMergedScene time spend:", spend_time_ms, "ms")
                continue
            if not self.dataset_render_manager.scannet_sim_loader.sim_manager.keyBoardControl(
                input_key
            ):
                break

        self.dataset_render_manager.scannet_sim_loader.sim_manager.cv_renderer.close()
        return True
