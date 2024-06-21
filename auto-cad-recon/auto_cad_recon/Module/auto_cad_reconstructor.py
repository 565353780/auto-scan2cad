#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
from time import time
from copy import deepcopy

import numpy as np
import open3d as o3d
import torch
from getch import getch
from global_pose_refine.Data.obb import OBB
from global_pose_refine.Module.detector import Detector as GPRDetector
from noc_transform.Module.transform_generator import TransformGenerator
from points_shape_detect.Loss.ious import IoULoss
from points_shape_detect.Method.matrix import getRotateMatrix
from points_shape_detect.Method.trans import getInverseTrans, transPointArray
from points_shape_detect.Module.detector import Detector as COBDetector
from points_shape_detect.Module.rotate_detector import RotateDetector
from td_r2n2.Module.detector import TDR2N2Detector
from tqdm import tqdm
from udf_generate.Method.udfs import getPointUDF

from auto_cad_recon.Method.scene_render import getMergedScenePCD
from auto_cad_recon.Method.bbox import getMoveToNOCTrans, getOBBFromABB
from auto_cad_recon.Method.device import toCpu, toCuda, toNumpy
from auto_cad_recon.Method.io import saveGTMeshInfo, saveAllRenderResult
from auto_cad_recon.Method.render import renderDataList, renderMergedScene
from auto_cad_recon.Method.trans import transPoints
from auto_cad_recon.Model.roca import RetrievalNet
from auto_cad_recon.Module.dataset_manager import DatasetManager
from auto_cad_recon.Module.dataset_render_manager import DatasetRenderManager
from auto_cad_recon.Module.nbv_generator import NBVGenerator
from auto_cad_recon.Module.retrieval_manager import RetrievalManager

fusion_mode_list = ["td_r2n2", "merge_points"]
fusion_mode = "merge_points"

retrieval_mode_list = [
    "roca",
    "conv_onet_encode",
    "conv_onet_occ",
    "occ",
    "occ_noise",
    "udf",
]
retrieval_mode = "occ"

trans_back_mode_List = ["gt", "bbox_net"]
trans_back_mode = "bbox_net"


class AutoCADReconstructor(object):
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

        if fusion_mode == "td_r2n2":
            model_file_path = "/home/chli/chLi/3D-R2N2/checkpoint.pth"
            self.fusion_detector = TDR2N2Detector(model_file_path)

        self.retrieval_manager = None
        if retrieval_mode == "roca":
            self.retrieval_net = RetrievalNet().cuda()
        else:
            self.retrieval_manager = RetrievalManager(
                retrieval_mode,
                scannet_dataset_folder_path,
                scannet_object_dataset_folder_path,
                scannet_bbox_dataset_folder_path,
                scan2cad_dataset_folder_path,
                scan2cad_object_model_map_dataset_folder_path,
                shapenet_dataset_folder_path,
                shapenet_udf_dataset_folder_path,
            )

        if trans_back_mode == "bbox_net":
            cob_model_file_path = "/home/chli/chLi/auto-scan2cad/my_model/points_shape_detect/output/pretrained_bbox1/model_best.pth"
            rot_model_file_path = "/home/chli/chLi/auto-scan2cad/my_model/points_shape_detect/output/pretrained_transformer_rotate1/model_best.pth"
            self.cob_detector = COBDetector(cob_model_file_path)
            self.rot_detector = RotateDetector(rot_model_file_path)

        gpr_model_file_path = "/home/chli/chLi/auto-scan2cad/my_model/global_pose_refine/output/pretrained_gcnn/model_best.pth"
        self.gpr_detector = GPRDetector(gpr_model_file_path)

        self.transform_generator = TransformGenerator()

        self.nbv_generator = NBVGenerator()

        self.save_render_result_idx = 0
        self.key_str = ""
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

    def loadRetrievalNetModel(self, model_file_path):
        assert os.path.exists(model_file_path)

        model_dict = torch.load(model_file_path)
        self.retrieval_net.load_state_dict(model_dict["retrieval_net"])
        return True

    def loadScene(self, scannet_scene_name, print_progress=False):
        self.scannet_scene_name = scannet_scene_name
        valid_object_file_name_list = self.getScanNetObjectFileNameList(
            self.scannet_scene_name
        )
        self.dataset_render_manager.loadScene(
            self.scannet_scene_name, print_progress, valid_object_file_name_list
        )

        if self.retrieval_manager is not None:
            self.retrieval_manager.loadScene(scannet_scene_name)
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

    def fusionSceneObjectByTDR2N2(self, data):
        bbox_image_width = 127
        bbox_image_height = 127
        bbox_image_free_width = 5

        assert (
            data["inputs"]["object_label"]
            in self.dataset_render_manager.scannet_sim_loader.scene_object_manager.scene_object_dict.keys()
        )

        scene_object = self.dataset_render_manager.scannet_sim_loader.scene_object_manager.scene_object_dict[
            data["inputs"]["object_label"]
        ]

        object_bbox_image_list = []
        for frame_object in scene_object.frame_object_dict.values():
            bbox_image = frame_object.getBBoxImage(
                bbox_image_width, bbox_image_height, bbox_image_free_width
            )
            object_bbox_image_list.append(bbox_image)

        fusion_data = self.fusion_detector.detectImages(object_bbox_image_list)
        for key in data.keys():
            data[key].update(fusion_data[key])

        for key, value in data["predictions"].items():
            try:
                print(key, value.shape)
            except:
                continue
        return data

    def fusionSceneObjectByMergePoints(self, data):
        assert (
            data["inputs"]["object_label"]
            in self.dataset_render_manager.scannet_sim_loader.scene_object_manager.scene_object_dict.keys()
        )

        scene_object = self.dataset_render_manager.scannet_sim_loader.scene_object_manager.scene_object_dict[
            data["inputs"]["object_label"]
        ]

        merged_point_array, merged_color_array = scene_object.getMergedPointArray()
        data["predictions"]["merged_point_array"] = merged_point_array
        data["predictions"]["merged_color_array"] = merged_color_array

        min_point_list = [np.min(merged_point_array[:, i]) for i in range(3)]
        max_point_list = [np.max(merged_point_array[:, i]) for i in range(3)]

        data["predictions"]["init_translate"] = torch.tensor(
            [[-(min_point_list[i] + max_point_list[i]) for i in range(3)]]
        ).type(torch.FloatTensor)

        diff_max = np.max([max_point_list[i] - min_point_list[i] for i in range(3)])

        if diff_max > 0:
            data["predictions"]["init_scale"] = torch.tensor(
                [[1.0 / diff_max for _ in range(3)]]
            ).type(torch.FloatTensor)
        else:
            data["predictions"]["init_scale"] = torch.tensor(
                [[1.0 for _ in range(3)]]
            ).type(torch.FloatTensor)

        point_udf = getPointUDF(merged_point_array)

        data["predictions"]["point_udf"] = torch.tensor(point_udf[None, :]).type(
            torch.FloatTensor
        )
        return data

    def fusionSceneObject(self, data):
        assert fusion_mode in fusion_mode_list

        if fusion_mode == "td_r2n2":
            return self.fusionSceneObjectByTDR2N2(data)
        if fusion_mode == "merge_points":
            return self.fusionSceneObjectByMergePoints(data)

    def getTransBackPointsByGT(self, data):
        merged_point_array = data["predictions"]["merged_point_array"]
        trans_matrix_inv = data["inputs"]["trans_matrix_inv"][0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_point_array)
        pcd.transform(trans_matrix_inv)

        points = np.array(pcd.points).astype(np.float32).reshape(1, -1, 3)

        data["predictions"]["trans_matrix"] = trans_matrix_inv.transpose(1, 0)
        data["predictions"]["trans_back_points"] = points
        return data

    def getTransBackPointsByBBoxNet(self, data):
        point_array = data["predictions"]["merged_point_array"]

        merged_point_array = data["predictions"]["merged_point_array"]
        trans_matrix_inv = data["inputs"]["trans_matrix_inv"][0]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_point_array)
        pcd.transform(trans_matrix_inv)
        gt_point_array = np.array(pcd.points).astype(np.float32)

        center = np.mean(point_array, axis=0)

        origin_point_array = point_array - center

        init_rotate_matrix = getRotateMatrix([90, 0, 0], True)
        rotate_origin_point_array = origin_point_array @ init_rotate_matrix
        rotate_data = self.rot_detector.detectPointArray(rotate_origin_point_array)
        rotate_matrix_inv = init_rotate_matrix @ rotate_data["predictions"][
            "rotate_matrix"
        ][0].detach().cpu().numpy().transpose(1, 0)
        rotate_matrix = rotate_matrix_inv.transpose(1, 0)

        rotate_back_point_array = origin_point_array @ rotate_matrix_inv

        cob_data = self.cob_detector.detectPointArray(rotate_back_point_array)
        origin_bbox = cob_data["predictions"]["origin_bbox"][0].detach().cpu().numpy()
        origin_center = (
            cob_data["predictions"]["origin_center"][0].detach().cpu().numpy()
        )

        translate, euler_angle, scale = getMoveToNOCTrans(origin_bbox, origin_center)
        points = transPointArray(
            rotate_back_point_array, translate, euler_angle, scale, is_inverse=True
        )
        noc_bbox = transPointArray(
            origin_bbox.reshape(2, 3), translate, euler_angle, scale, is_inverse=True
        ).reshape(-1)
        noc_center = origin_center + translate

        points = points.astype(np.float32).reshape(1, -1, 3)

        data["predictions"]["center"] = center
        data["predictions"]["rotate_matrix"] = rotate_matrix
        data["predictions"]["rotate_matrix_inv"] = rotate_matrix_inv
        data["predictions"]["rotate_back_point_array"] = rotate_back_point_array
        data["predictions"]["origin_bbox"] = origin_bbox
        data["predictions"]["origin_center"] = origin_center
        data["predictions"]["noc_translate"] = translate
        data["predictions"]["noc_euler_angle"] = euler_angle
        data["predictions"]["noc_scale"] = scale
        data["predictions"]["noc_bbox"] = noc_bbox
        data["predictions"]["noc_center"] = noc_center

        data["predictions"]["rotate_origin_point_array"] = rotate_origin_point_array
        data["predictions"]["trans_back_points"] = points
        data["predictions"]["refine_trans_back_points"] = points
        data["inputs"]["gt_trans_back_points"] = gt_point_array
        return data

    def getTransBackPoints(self, data):
        if trans_back_mode == "gt":
            return self.getTransBackPointsByGT(data)
        else:
            return self.getTransBackPointsByBBoxNet(data)

    def globalPoseRefine(self, data_list, layout_data):
        wall_height = 3

        floor_position = np.array(layout_data["predictions"]["floor_array"])
        floor_position = np.array([[0, 0, 0], [1, 1, 0]], dtype=float)
        floor_normal = np.array(
            [[0.0, 0.0, 1.0] for _ in range(floor_position.shape[0])]
        )
        floor_z_value = np.array([[0.0] for _ in range(floor_position.shape[0])])
        floor_abb = np.hstack([floor_position, floor_position]) + [
            -1000,
            -1000,
            -0.01,
            1000,
            1000,
            0,
        ]
        floor_obb = np.array([OBB.fromABBList(abb).toArray() for abb in floor_abb])

        wall_position_list = []
        wall_normal_list = []
        for i in range(floor_position.shape[0]):
            start_idx = i
            end_idx = (i + 1) % floor_position.shape[0]

            wall_position = np.array(
                [
                    floor_position[start_idx],
                    floor_position[end_idx],
                    floor_position[end_idx] + [0.0, 0.0, wall_height],
                    floor_position[start_idx] + [0.0, 0.0, wall_height],
                ]
            )
            wall_diff = wall_position[1] - wall_position[0]
            wall_normal = np.array([-wall_diff[1], wall_diff[0], 0])
            wall_normal = wall_normal / np.linalg.norm(wall_normal)

            wall_position_list.append(wall_position)
            wall_normal_list.append(wall_normal)

        wall_position = np.array(wall_position_list)
        wall_normal = np.array(wall_normal_list)
        wall_obb = np.hstack([wall_position, wall_position])
        wall_obb[:, :4, :] -= wall_normal.reshape(-1, 1, 3) * 0.01
        wall_obb[:, 4:, :] += wall_normal.reshape(-1, 1, 3) * 0.01

        object_obb_list = []
        object_abb_list = []
        object_obb_center_list = []
        for data in data_list:
            center = data["predictions"]["center"]
            rotate_matrix = data["predictions"]["rotate_matrix"]
            noc_translate = data["predictions"]["noc_translate"]
            noc_euler_angle = data["predictions"]["noc_euler_angle"]
            noc_scale = data["predictions"]["noc_scale"]
            noc_bbox = data["predictions"]["noc_bbox"]
            merged_point_array = data["predictions"]["merged_point_array"]

            object_obb = getOBBFromABB(noc_bbox)

            noc_translate_inv, noc_euler_angle_inv, noc_scale_inv = getInverseTrans(
                noc_translate, noc_euler_angle, noc_scale
            )

            object_obb = transPointArray(
                object_obb, noc_translate_inv, noc_euler_angle_inv, noc_scale_inv
            )

            object_obb = object_obb @ rotate_matrix

            object_obb = object_obb + center

            object_abb = np.hstack(
                (np.min(object_obb, axis=0), np.max(object_obb, axis=0))
            )

            object_obb_center = np.mean(object_obb, axis=0)

            object_obb_list.append(object_obb)
            object_abb_list.append(object_abb)
            object_obb_center_list.append(object_obb_center)

        object_obb = np.array(object_obb_list)
        object_abb = np.array(object_abb_list)
        object_obb_center = np.array(object_obb_center_list)

        obb_list = np.vstack([object_obb, wall_obb, floor_obb])
        obb_center_list = np.array(
            [np.mean(OBB(obb).points, axis=0) for obb in obb_list]
        )

        obb_center_dist_list = [
            np.linalg.norm(center2 - center1, ord=2)
            for center1 in obb_center_list
            for center2 in obb_center_list
        ]

        object_abb = torch.from_numpy(object_abb)

        abb_list = np.array([OBB(obb).toABBArray() for obb in obb_list])
        abb_list = torch.from_numpy(abb_list).float().reshape(-1, 6)

        abb_eiou_list = [
            IoULoss.EIoU(bbox1, bbox2) for bbox1 in abb_list for bbox2 in abb_list
        ]

        obb_center_dist = torch.tensor(obb_center_dist_list)
        abb_eiou = torch.tensor(abb_eiou_list)

        gpr_data = {"inputs": {}, "predictions": {}, "losses": {}, "logs": {}}

        gpr_data["inputs"]["floor_position"] = torch.from_numpy(floor_position)
        gpr_data["inputs"]["floor_normal"] = torch.from_numpy(floor_normal)
        gpr_data["inputs"]["floor_z_value"] = torch.from_numpy(floor_z_value)

        gpr_data["inputs"]["wall_position"] = torch.from_numpy(wall_position)
        gpr_data["inputs"]["wall_normal"] = torch.from_numpy(wall_normal)

        gpr_data["inputs"]["trans_object_obb"] = torch.from_numpy(object_obb)
        gpr_data["inputs"]["trans_object_abb"] = object_abb
        gpr_data["inputs"]["trans_object_obb_center"] = torch.from_numpy(
            object_obb_center
        )

        gpr_data["inputs"]["trans_obb_center_dist"] = obb_center_dist
        gpr_data["inputs"]["trans_abb_eiou"] = abb_eiou

        gpr_data = self.gpr_detector.detectSceneTrans(gpr_data)

        refine_translate_inv = gpr_data["predictions"]["refine_translate_inv"][0]
        refine_rotate_matrix_inv = gpr_data["predictions"]["refine_rotate_matrix_inv"][
            0
        ]
        refine_scale_inv = gpr_data["predictions"]["refine_scale_inv"][0]
        refine_object_obb = gpr_data["predictions"]["refine_object_obb"][0]

        relation_matrix = gpr_data["predictions"]["relation_matrix"]

        for i in range(len(data_list)):
            data_list[i]["predictions"]["refine_translate_inv"] = refine_translate_inv[
                i
            ]
            data_list[i]["predictions"][
                "refine_rotate_matrix_inv"
            ] = refine_rotate_matrix_inv[i]
            data_list[i]["predictions"]["refine_scale_inv"] = refine_scale_inv[i]
            data_list[i]["predictions"]["refine_obb"] = refine_object_obb[i].reshape(
                8, 3
            )

            obb = self.transform_generator.getOBB(
                data_list[i]["predictions"]["refine_obb"]
            )

            refine_noc_obb = self.transform_generator.getNOCOBB(obb).toArray()
            refine_transform_inv = self.transform_generator.getNOCTransform(obb)
            refine_transform = np.linalg.inv(refine_transform_inv)

            merged_point_array = data_list[i]["predictions"]["merged_point_array"]

            refine_trans_back_points = transPoints(
                merged_point_array, refine_transform_inv
            ).reshape(1, -1, 3)

            data_list[i]["predictions"]["refine_noc_obb"] = refine_noc_obb
            data_list[i]["predictions"]["refine_transform"] = refine_transform
            data_list[i]["predictions"][
                "refine_trans_back_points"
            ] = refine_trans_back_points

        object_num = object_obb.shape[0]
        wall_num = wall_position.shape[0]
        floor_num = floor_position.shape[0]
        total_num = object_num + wall_num + floor_num

        relation_matrix = relation_matrix.reshape((total_num, total_num))

        relation_data = {
            "object_num": object_num,
            "wall_num": wall_num,
            "floor_num": floor_num,
            "relation_matrix": relation_matrix,
        }
        return data_list, relation_data

    def retrievalSceneObjectByROCA(self, data):
        toCuda(data)
        data = self.retrieval_net(data)
        toCpu(data)
        toNumpy(data)
        return data

    def retrievalSceneObjectByManager(self, data):
        points = data["predictions"]["refine_trans_back_points"]

        (
            retrieval_model_file_path,
            retrieval_object_file_name,
        ) = self.retrieval_manager.getPointArrayRetrievalResult(points)

        #  retrieval_model_file_path, retrieval_object_file_name = \
        #  self.retrieval_manager.getPointArrayRetrievalResultWithICP(points)

        data["predictions"]["retrieval_model_file_path"] = retrieval_model_file_path
        data["predictions"]["retrieval_object_file_name"] = retrieval_object_file_name
        return data

    def retrievalSceneObject(self, data):
        assert retrieval_mode in retrieval_mode_list

        if retrieval_mode == "roca":
            return self.retrievalSceneObjectByROCA(data)
        else:
            return self.retrievalSceneObjectByManager(data)

    def generateNBV(self, data_list, relation_data):
        # TODO: code need to be simplified into single func, current release version will use manual control as default
        nbv_list = self.nbv_generator.generateNBV(data_list, relation_data)
        return True

    def processAllSceneObjects(self):
        start_time = time()

        data_list = []

        layout_data = {"inputs": {}, "predictions": {}, "losses": {}, "logs": {}}
        layout_data["predictions"][
            "floor_array"
        ] = self.dataset_render_manager.scannet_sim_loader.layout_map_builder.layout_map.floor_array
        layout_data["predictions"][
            "layout_mesh"
        ] = self.dataset_render_manager.scannet_sim_loader.layout_map_builder.layout_mesh

        for object_label in self.dataset_render_manager.scannet_sim_loader.scene_object_manager.scene_object_dict.keys():
            if "==object" not in object_label:
                continue

            scannet_object_file_name = object_label.split("==object")[0]
            if not self.isObjectValid(
                self.scannet_scene_name, scannet_object_file_name
            ):
                continue

            data = {"inputs": {}, "predictions": {}, "losses": {}, "logs": {}}
            data["inputs"]["dataset_manager"] = self.dataset_manager
            data["inputs"]["scannet_scene_name"] = self.scannet_scene_name
            data["inputs"]["object_label"] = object_label
            data["inputs"]["scannet_object_file_name"] = scannet_object_file_name
            data["inputs"].update(
                self.getShapeNetModelTensorDict(
                    self.scannet_scene_name, data["inputs"]["scannet_object_file_name"]
                )
            )
            data_list.append(data)

        for i in range(len(data_list)):
            data_list[i] = self.fusionSceneObject(data_list[i])

        for i in range(len(data_list)):
            data_list[i] = self.getTransBackPoints(data_list[i])

        relation_data = None
        if len(data_list) > 0:
            data_list, relation_data = self.globalPoseRefine(data_list, layout_data)

        for i in range(len(data_list)):
            data_list[i] = self.retrievalSceneObject(data_list[i])

        if fusion_mode == "td_r2n2":
            for data in data_list:
                self.fusion_detector.saveAsObj(
                    data,
                    "./output/scene/"
                    + self.scannet_scene_name
                    + "/fusion/"
                    + scannet_object_file_name.split(".")[0]
                    + ".obj",
                )

        spend_time_ms = 1000 * (time() - start_time)
        print("infer spend time:", spend_time_ms, "ms")

        self.generateNBV(data_list, relation_data)
        # return True

        scene_points_list = []
        scene_colors_list = []
        for (
            point_image
        ) in self.dataset_render_manager.scannet_sim_loader.point_image_list:
            point_idx = np.where(point_image.point_array[:, 0] != float("inf"))[0]

            points = point_image.point_array[point_idx]
            colors = point_image.image.reshape(-1, 3)[..., [2, 1, 0]][point_idx]
            scene_points_list.append(points)
            scene_colors_list.append(colors)
        scene_points_array = np.vstack(scene_points_list)
        scene_colors_array = np.vstack(scene_colors_list)

        save_folder_path = (
            "./output/scene/"
            + self.scannet_scene_name
            + "/frames/"
            + str(self.save_render_result_idx)
            + "/"
        )

        saveAllRenderResult(
            save_folder_path,
            data_list,
            layout_data,
            self.dataset_render_manager.scannet_sim_loader.point_image_list,
            "scan",
            "gt",
            "gt",
            is_scene_gray=False,
            print_progress=True,
        )

        saveAllRenderResult(
            save_folder_path + "retrieval/",
            data_list,
            layout_data,
            self.dataset_render_manager.scannet_sim_loader.point_image_list,
            "scan",
            "gt",
            "retrieval",
            is_scene_gray=False,
            print_progress=True,
        )

        save_folder_path = (
            "./output/scene/"
            + self.scannet_scene_name
            + "/mesh_info/"
            + str(self.save_render_result_idx)
            + "/"
        )

        saveGTMeshInfo(save_folder_path, data_list)

        self.save_render_result_idx += 1

        return True
        renderDataList(
            data_list,
            layout_data,
            self.dataset_render_manager.scannet_sim_loader.point_image_list,
        )
        return True

    def renderMergedScene(self):
        renderMergedScene(
            self.dataset_render_manager.scannet_sim_loader.point_image_list,
            background_only=False,
            is_gray=False,
            estimate_normals=True,
        )

        keys_save_folder_path = "./output/keys/" + self.scannet_scene_name + "/"
        os.makedirs(keys_save_folder_path, exist_ok=True)
        with open(keys_save_folder_path + "keys.txt", "w") as f:
            f.write(self.key_str)
        return True

    def renderSceneObject(self):
        scene_object_dict = self.dataset_render_manager.scannet_sim_loader.scene_object_manager.scene_object_dict

        for object_label, scene_object in scene_object_dict.items():
            merged_point_array, merged_color_array = scene_object.getMergedPointArray()
            object_file_name = object_label.split("==object")[0]

            print(object_label, merged_point_array.shape)
        return True

    def saveCurrentObservation(self, render=False):
        point_image_list = (
            self.dataset_render_manager.scannet_sim_loader.point_image_list
        )
        explore_map = self.dataset_render_manager.scannet_sim_loader.layout_map_builder.explore_map

        root_save_folder_path = "./output/scene/" + self.scannet_scene_name

        observation_save_folder_path = root_save_folder_path + "/observation/"
        os.makedirs(observation_save_folder_path, exist_ok=True)
        explore_map_save_folder_path = root_save_folder_path + "/explore_map/"
        os.makedirs(explore_map_save_folder_path, exist_ok=True)
        camera_position_save_folder_path = root_save_folder_path + "/camera_position/"
        os.makedirs(camera_position_save_folder_path, exist_ok=True)
        rgbd_save_folder_path = root_save_folder_path + "/rgbd/"
        os.makedirs(rgbd_save_folder_path, exist_ok=True)
        semantic_save_folder_path = root_save_folder_path + "/semantic/"
        os.makedirs(semantic_save_folder_path, exist_ok=True)

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
        semantic = point_image_list[-1].getAllLabelRender()
        cv2.imwrite(
            semantic_save_folder_path + current_idx_str + "_semantic.png", semantic
        )

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

        root_save_folder_path = "./output/scene/" + self.scannet_scene_name

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
            self.key_str += input_key
            if input_key == "v":
                start_time = time()
                self.dataset_render_manager.scannet_sim_loader.getObjectInView(
                    print_progress
                )
                spend_time_ms = 1000 * (time() - start_time)
                print("fusion time spend:", spend_time_ms, "ms")
                self.renderMergedScene()
                continue
            if input_key == "x":
                self.processAllSceneObjects()
                continue
            if input_key == "t":
                self.renderSceneObject()
                continue
            if input_key == "n":
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
                start_time = time()
                self.saveMergedScene()
                spend_time_ms = 1000 * (time() - start_time)
                print("saveMergedScene time spend:", spend_time_ms, "ms")
                start_time = time()
                self.processAllSceneObjects()
                spend_time_ms = 1000 * (time() - start_time)
                print("processAllSceneObjects time spend:", spend_time_ms, "ms")
                continue
            if not self.dataset_render_manager.scannet_sim_loader.sim_manager.keyBoardControl(
                input_key
            ):
                break

        self.dataset_render_manager.scannet_sim_loader.sim_manager.cv_renderer.close()
        return True
