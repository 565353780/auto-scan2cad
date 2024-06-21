#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from copy import deepcopy

import numpy as np
import open3d as o3d
from conv_onet.Data.crop_space import CropSpace

#  from conv_onet.Module.detector import Detector as ConvONetDetector
from mesh_manage.Method.heatmap import getNoise
from tqdm import tqdm
from udf_generate.Method.udfs import getPointUDF

from points_shape_detect.Method.trans import normalizePointArray

from auto_cad_recon.Method.match_check import isMatch
from auto_cad_recon.Method.path import createFileFolder
from auto_cad_recon.Module.dataset_manager import DatasetManager


def draw_registration_result(source, target):
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    o3d.visualization.draw_geometries([source_temp, target_temp])
    return True


class RetrievalManager(object):
    def __init__(
        self,
        mode,
        scannet_dataset_folder_path,
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

        self.mode_list = [
            "conv_onet_encode",
            "conv_onet_occ",
            "occ",
            "occ_noise",
            "udf",
        ]
        self.mode = mode
        assert self.mode in self.mode_list

        self.conv_onet_detector = None
        self.crop_space = None
        if "conv_onet" in self.mode:
            self.conv_onet_detector = ConvONetDetector()
        else:
            self.crop_space = CropSpace(0.1, 0.1, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])

        self.scannet_scene_name = None
        self.object_file_name_list = []
        self.object_points_list = []
        self.trans_matrix_list = []
        self.cad_file_path_list = []

        self.cad_feature_list = []
        self.cad_mask_list = []
        self.cad_valid_num_list = []

        self.cad_pcd_list = []
        return

    def reset(self):
        self.scannet_scene_name = None
        self.object_file_name_list = []
        self.object_points_list = []
        self.trans_matrix_list = []
        self.cad_file_path_list = []

        self.cad_feature_list = []
        self.cad_mask_list = []
        self.cad_valid_num_list = []

        self.cad_pcd_list = []
        return True

    def loadScene(self, scannet_scene_name, preload_feature=True):
        if self.scannet_scene_name == scannet_scene_name:
            return True

        self.reset()

        self.scannet_scene_name = scannet_scene_name
        self.object_file_name_list = self.dataset_manager.getScanNetObjectFileNameList(
            scannet_scene_name
        )

        self.object_points_list = []
        self.trans_matrix_list = []
        self.cad_file_path_list = []
        for object_file_name in self.object_file_name_list:
            shapenet_model_dict = self.dataset_manager.getShapeNetModelDict(
                scannet_scene_name, object_file_name
            )
            object_file_path = shapenet_model_dict["scannet_object_file_path"]
            cad_file_path = shapenet_model_dict["shapenet_model_file_path"]
            trans_matrix = shapenet_model_dict["trans_matrix"]
            trans_matrix_inv = shapenet_model_dict["trans_matrix_inv"]

            pcd = o3d.io.read_point_cloud(object_file_path)
            pcd.transform(trans_matrix_inv)

            points = np.array(pcd.points)
            points = normalizePointArray(points)

            points = points.astype(np.float32).reshape(1, -1, 3)
            self.object_points_list.append(points)
            self.trans_matrix_list.append(trans_matrix)
            self.cad_file_path_list.append(cad_file_path)

        if preload_feature:
            self.getAllCADFeaturesAndMasks()
        return True

    def getFeatureAndMask(self, points):
        assert self.mode in self.mode_list

        if self.mode == "conv_onet_encode":
            cad_result = self.conv_onet_detector.detect(points)
            feature = cad_result["encode"]
            mask = cad_result["mask"]
            return feature, mask
        if self.mode == "conv_onet_occ":
            cad_result = self.conv_onet_detector.detect(points)
            feature = cad_result["occ"]
            mask = cad_result["mask"]
            return feature, mask
        if self.mode == "occ":
            self.crop_space.updatePointArray(points)
            feature = self.crop_space.getFeatureArray("valid")
            mask = self.crop_space.getFeatureMaskArray("valid")
            return feature, mask
        if self.mode == "occ_noise":
            noise = getNoise(points.shape[0], 0.01)
            points += noise
            self.crop_space.updatePointArray(points)
            feature = self.crop_space.getFeatureArray("valid")
            mask = self.crop_space.getFeatureMaskArray("valid")
            return feature, mask
        if self.mode == "udf":
            udf_size = 10
            feature = getPointUDF(points.reshape(-1, 3)).reshape(
                udf_size, udf_size, udf_size, 1
            )
            mask = np.ones([udf_size, udf_size, udf_size], dtype=bool)
            return feature, mask

    def getAllCADFeaturesAndMasks(self):
        if "conv_onet" in self.mode:
            print("[WARN][RetrievalManager::getAllCADFeaturesAndMasks]")
            print("\t conv_onet feature could not save into RAM! will skip this!")
            return True

        for i, cad_file_path in enumerate(self.cad_file_path_list):
            print("[INFO][RetrievalManager::getAllCADFeaturesAndMasks]")
            print(
                "\t get features for cad "
                + str(i + 1)
                + "/"
                + str(len(self.cad_file_path_list))
                + "..."
            )

            mesh = o3d.io.read_triangle_mesh(cad_file_path)
            pcd = mesh.sample_points_uniformly(100000)
            points = np.array(pcd.points, dtype=np.float32)

            points = normalizePointArray(points)

            points = points.reshape(1, -1, 3)

            cad_source_feature, cad_mask = self.getFeatureAndMask(points)
            cad_valid_num = np.where(cad_mask == True)[0].shape[0]
            self.cad_feature_list.append(cad_source_feature)
            self.cad_mask_list.append(cad_mask)
            self.cad_valid_num_list.append(cad_valid_num)

            self.cad_pcd_list.append(pcd)
        return True

    def getPointArrayRetrievalResult(self, point_array):
        error_list = []

        object_source_feature, mask = self.getFeatureAndMask(point_array)

        for cad_source_feature, cad_mask, cad_valid_num in zip(
            self.cad_feature_list, self.cad_mask_list, self.cad_valid_num_list
        ):
            merge_mask = cad_mask & mask
            merge_feature_idx = np.dstack(np.where(merge_mask == True))[0]

            merge_error = 0

            for j, k, l in merge_feature_idx:
                cad_feature = cad_source_feature[j, k, l].flatten()
                object_feature = object_source_feature[j, k, l].flatten()
                merge_error += np.linalg.norm(cad_feature - object_feature, ord=2)

            if len(merge_feature_idx) > 0:
                merge_error /= len(merge_feature_idx)

            object_error = 0

            object_only_mask = ~cad_mask & mask
            object_only_feature_idx = np.dstack(np.where(object_only_mask == True))[0]
            for j, k, l in object_only_feature_idx:
                object_feature = object_source_feature[j, k, l].flatten()
                object_error += np.linalg.norm(object_feature, ord=2)

            object_valid_num = np.where(mask == True)[0].shape[0]
            if object_valid_num > 0:
                object_weight = object_only_feature_idx.shape[0] / object_valid_num
            else:
                object_weight = 0
            object_error *= object_weight

            cad_error = 0

            cad_only_mask = cad_mask & ~mask
            cad_only_feature_idx = np.dstack(np.where(cad_only_mask == True))[0]
            for j, k, l in cad_only_feature_idx:
                cad_feature = cad_source_feature[j, k, l].flatten()
                cad_error += np.linalg.norm(cad_feature, ord=2)

            if cad_valid_num > 0:
                cad_weight = cad_only_feature_idx.shape[0] / cad_valid_num
            else:
                cad_weight = 0
            cad_error *= cad_weight

            error = merge_error + 0.8 * object_error + 0.2 * cad_error

            error_list.append(error)

        min_error_idx = np.argmin(error_list)
        min_error_cad_model_file_path = self.cad_file_path_list[min_error_idx]
        min_error_object_file_name = self.object_file_name_list[min_error_idx]
        return min_error_cad_model_file_path, min_error_object_file_name

    def getPointArrayRetrievalResultWithICP(self, point_array):
        error_list = []

        for cad_source_feature, cad_mask, cad_valid_num, cad_pcd in zip(
            self.cad_feature_list,
            self.cad_mask_list,
            self.cad_valid_num_list,
            self.cad_pcd_list,
        ):
            icp_point_array = deepcopy(point_array)[0]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(icp_point_array)

            reg_p2l = o3d.pipelines.registration.registration_icp(
                pcd,
                cad_pcd,
                0.02,
                np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )

            pcd.transform(reg_p2l.transformation)
            #  draw_registration_result(pcd, cad_pcd)

            icp_point_array = np.array(pcd.points).reshape(1, -1, 3)

            object_source_feature, mask = self.getFeatureAndMask(icp_point_array)

            merge_mask = cad_mask & mask
            merge_feature_idx = np.dstack(np.where(merge_mask == True))[0]

            merge_error = 0

            for j, k, l in merge_feature_idx:
                cad_feature = cad_source_feature[j, k, l].flatten()
                object_feature = object_source_feature[j, k, l].flatten()
                merge_error += np.linalg.norm(cad_feature - object_feature, ord=2)

            if len(merge_feature_idx) > 0:
                merge_error /= len(merge_feature_idx)

            object_error = 0

            object_only_mask = ~cad_mask & mask
            object_only_feature_idx = np.dstack(np.where(object_only_mask == True))[0]
            for j, k, l in object_only_feature_idx:
                object_feature = object_source_feature[j, k, l].flatten()
                object_error += np.linalg.norm(object_feature, ord=2)

            object_valid_num = np.where(mask == True)[0].shape[0]
            if object_valid_num > 0:
                object_weight = object_only_feature_idx.shape[0] / object_valid_num
            else:
                object_weight = 0
            object_error *= object_weight

            cad_error = 0

            cad_only_mask = cad_mask & ~mask
            cad_only_feature_idx = np.dstack(np.where(cad_only_mask == True))[0]
            for j, k, l in cad_only_feature_idx:
                cad_feature = cad_source_feature[j, k, l].flatten()
                cad_error += np.linalg.norm(cad_feature, ord=2)

            if cad_valid_num > 0:
                cad_weight = cad_only_feature_idx.shape[0] / cad_valid_num
            else:
                cad_weight = 0
            cad_error *= cad_weight

            error = merge_error + 0.8 * object_error + 0.2 * cad_error

            error_list.append(error)

        min_error_idx = np.argmin(error_list)
        min_error_cad_model_file_path = self.cad_file_path_list[min_error_idx]
        min_error_object_file_name = self.object_file_name_list[min_error_idx]
        return min_error_cad_model_file_path, min_error_object_file_name

    def getErrorMatrix(self):
        cad_to_scan_errors_list = []

        for i, cad_file_path in enumerate(self.cad_file_path_list):
            print(
                "get error for cad "
                + str(i + 1)
                + "/"
                + str(len(self.cad_file_path_list))
                + "..."
            )

            if len(self.cad_mask_list) == 0:
                mesh = o3d.io.read_triangle_mesh(cad_file_path)
                pcd = mesh.sample_points_uniformly(1000000)
                points = np.array(pcd.points, dtype=np.float32).reshape(1, -1, 3)

                cad_source_feature, cad_mask = self.getFeatureAndMask(points)
            else:
                cad_source_feature = self.cad_feature_list[i]
                cad_mask = self.cad_mask_list[i]
            cad_valid_num = np.where(cad_mask == True)[0].shape[0]

            cad_to_scan_errors = []
            for points in tqdm(self.object_points_list):
                object_source_feature, mask = self.getFeatureAndMask(points)

                merge_mask = cad_mask & mask
                merge_feature_idx = np.dstack(np.where(merge_mask == True))[0]

                merge_error = 0

                for j, k, l in merge_feature_idx:
                    cad_feature = cad_source_feature[j, k, l].flatten()
                    object_feature = object_source_feature[j, k, l].flatten()
                    merge_error += np.linalg.norm(cad_feature - object_feature, ord=2)

                if len(merge_feature_idx) > 0:
                    merge_error /= len(merge_feature_idx)

                object_error = 0

                object_only_mask = ~cad_mask & mask
                object_only_feature_idx = np.dstack(np.where(object_only_mask == True))[
                    0
                ]
                for j, k, l in object_only_feature_idx:
                    object_feature = object_source_feature[j, k, l].flatten()
                    object_error += np.linalg.norm(object_feature, ord=2)

                object_valid_num = np.where(mask == True)[0].shape[0]
                if object_valid_num > 0:
                    object_weight = object_only_feature_idx.shape[0] / object_valid_num
                else:
                    object_weight = 0
                object_error *= object_weight

                cad_error = 0

                cad_only_mask = cad_mask & ~mask
                cad_only_feature_idx = np.dstack(np.where(cad_only_mask == True))[0]
                for j, k, l in cad_only_feature_idx:
                    cad_feature = cad_source_feature[j, k, l].flatten()
                    cad_error += np.linalg.norm(cad_feature, ord=2)

                if cad_valid_num > 0:
                    cad_weight = cad_only_feature_idx.shape[0] / cad_valid_num
                else:
                    cad_weight = 0
                cad_error *= cad_weight

                error = merge_error + 0.8 * object_error + 0.2 * cad_error

                cad_to_scan_errors.append(error)

            cad_to_scan_errors_list.append(cad_to_scan_errors)

        error_matrix = np.array(cad_to_scan_errors_list)
        scan_to_cad_error_matrix = error_matrix.T
        return scan_to_cad_error_matrix

    def generateErrorMatrix(self, save_pkl_file_path):
        createFileFolder(save_pkl_file_path)

        scan_to_cad_error_matrix = self.getErrorMatrix()

        pickle.dump(scan_to_cad_error_matrix, open(save_pkl_file_path, "wb"))
        return True

    def renderRetrievalResult(self, error_matrix_pkl_file_path):
        scan_to_cad_error_matrix = pickle.load(open(error_matrix_pkl_file_path, "rb"))

        object_file_name_list = self.dataset_manager.getScanNetObjectFileNameList(
            self.scannet_scene_name
        )

        object_pcd_list = []
        trans_matrix_list = []
        cad_file_path_list = []
        for object_file_name in object_file_name_list:
            shapenet_model_dict = self.dataset_manager.getShapeNetModelDict(
                self.scannet_scene_name, object_file_name
            )
            object_file_path = shapenet_model_dict["scannet_object_file_path"]
            trans_matrix = shapenet_model_dict["trans_matrix"]
            cad_file_path = shapenet_model_dict["shapenet_model_file_path"]

            pcd = o3d.io.read_point_cloud(object_file_path)
            object_pcd_list.append(pcd)

            trans_matrix_list.append(trans_matrix)
            cad_file_path_list.append(cad_file_path)

        cad_mesh_list = []

        error_max = 0.0
        for i, scan_to_cad_errors in enumerate(scan_to_cad_error_matrix):
            min_error_cad_idx = np.argmin(scan_to_cad_errors)
            min_error_object_file_name = object_file_name_list[min_error_cad_idx]

            mesh = o3d.io.read_triangle_mesh(cad_file_path_list[min_error_cad_idx])

            is_match = isMatch(min_error_object_file_name, object_file_name_list[i])
            is_error_min = scan_to_cad_errors[min_error_cad_idx] < error_max
            if is_match or is_error_min:
                mesh.paint_uniform_color(np.array([0.0, 1.0, 0.0]))
            else:
                mesh.paint_uniform_color(np.array([1.0, 0.0, 0.0]))

            mesh.transform(trans_matrix_list[i])
            mesh.compute_vertex_normals()
            cad_mesh_list.append(mesh)

        error_mesh_gt_list = []

        for i, scan_to_cad_errors in enumerate(scan_to_cad_error_matrix):
            min_error_cad_idx = np.argmin(scan_to_cad_errors)
            min_error_object_file_name = object_file_name_list[min_error_cad_idx]
            if not isMatch(min_error_object_file_name, object_file_name_list[i]):
                mesh = o3d.io.read_triangle_mesh(cad_file_path_list[i])
                mesh.paint_uniform_color(np.array([0.0, 0.0, 1.0]))
                mesh.transform(trans_matrix_list[i])
                mesh.compute_vertex_normals()
                error_mesh_gt_list.append(mesh)

        print(scan_to_cad_error_matrix)

        print("====Error Sum====")
        for i, scan_to_cad_errors in enumerate(scan_to_cad_error_matrix):
            min_error_cad_idx = np.argmin(scan_to_cad_errors)
            min_error_object_file_name = object_file_name_list[min_error_cad_idx]
            print("error =", scan_to_cad_errors[min_error_cad_idx], "\t", end="")
            print(
                object_file_name_list[i], "\t", "->", "\t", min_error_object_file_name
            )

        print("====Failed Case====")
        for i, scan_to_cad_errors in enumerate(scan_to_cad_error_matrix):
            min_error_cad_idx = np.argmin(scan_to_cad_errors)
            min_error_object_file_name = object_file_name_list[min_error_cad_idx]
            if not isMatch(min_error_object_file_name, object_file_name_list[i]):
                print("error =", scan_to_cad_errors[min_error_cad_idx], "\t", end="")
                print(
                    object_file_name_list[i],
                    "\t",
                    "->",
                    "\t",
                    min_error_object_file_name,
                )

        o3d.visualization.draw_geometries(
            object_pcd_list + cad_mesh_list + error_mesh_gt_list
        )
        return True
