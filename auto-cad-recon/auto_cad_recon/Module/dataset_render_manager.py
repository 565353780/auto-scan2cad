#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from getch import getch

from scannet_sim_manage.Module.scannet_sim_loader import ScanNetSimLoader


class DatasetRenderManager(object):

    def __init__(self, scannet_glb_dataset_folder_path,
                 scannet_object_dataset_folder_path,
                 scannet_bbox_dataset_folder_path):
        self.scannet_glb_dataset_folder_path = scannet_glb_dataset_folder_path
        self.scannet_object_dataset_folder_path = scannet_object_dataset_folder_path
        self.scannet_bbox_dataset_folder_path = scannet_bbox_dataset_folder_path

        self.scannet_sim_loader = ScanNetSimLoader()
        return

    def reset(self):
        self.scannet_sim_loader.reset()
        return True

    def loadScene(self,
                  scannet_scene_name,
                  print_progress=False,
                  valid_object_file_name_list=None):
        scannet_scene_glb_file_path = self.scannet_glb_dataset_folder_path + \
            scannet_scene_name + "/" + scannet_scene_name + "_vh_clean.glb"
        assert os.path.exists(scannet_scene_glb_file_path)

        scannet_scene_object_folder_path = self.scannet_object_dataset_folder_path + \
            scannet_scene_name + "/"
        assert os.path.exists(scannet_scene_object_folder_path)

        scannet_scene_bbox_file_path = self.scannet_bbox_dataset_folder_path + \
            scannet_scene_name + "/object_bbox.json"
        assert os.path.exists(scannet_scene_bbox_file_path)

        self.scannet_sim_loader.loadScene(scannet_scene_glb_file_path,
                                          scannet_scene_object_folder_path,
                                          scannet_scene_bbox_file_path,
                                          print_progress,
                                          valid_object_file_name_list)
        return True

    def setControlMode(self, control_mode):
        return self.scannet_sim_loader.setControlMode(control_mode)

    def setAgentPose(self, xyz_list, urf_list):
        return self.scannet_sim_loader.setAgentPose(xyz_list, urf_list)

    def getSceneObjectLabelList(self):
        return self.scannet_sim_loader.scene_object_manager.getSceneObjectLabelList(
        )

    def getFrameObjectDict(self, object_label):
        return self.scannet_sim_loader.scene_object_manager.getFrameObjectDict(
            object_label)

    def saveAllSceneObjects(self,
                            scene_objects_save_folder_path,
                            bbox_image_width=224,
                            bbox_image_height=224,
                            bbox_image_free_width=5):
        return self.scannet_sim_loader.saveAllSceneObjects(
            scene_objects_save_folder_path, bbox_image_width,
            bbox_image_height, bbox_image_free_width)

    def startKeyBoardControlRender(self, wait_val, print_progress=False):
        #  self.scannet_sim_loader.sim_manager.resetAgentPose()
        self.scannet_sim_loader.sim_manager.cv_renderer.init()

        while True:
            if not self.scannet_sim_loader.sim_manager.cv_renderer.renderFrame(
                    self.scannet_sim_loader.sim_manager.sim_loader.observations
            ):
                break
            self.scannet_sim_loader.sim_manager.cv_renderer.waitKey(wait_val)

            agent_state = self.scannet_sim_loader.sim_manager.sim_loader.getAgentState(
            )
            print("agent_state: position", agent_state.position, "rotation",
                  agent_state.rotation)

            input_key = getch()
            if input_key == "v":
                self.scannet_sim_loader.getObjectInView(print_progress)
                continue
            if not self.scannet_sim_loader.sim_manager.keyBoardControl(
                    input_key):
                break

        self.scannet_sim_loader.sim_manager.cv_renderer.close()
        return True
