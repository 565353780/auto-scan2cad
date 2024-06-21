#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pickle
from copy import deepcopy

import numpy as np
import open3d as o3d
from noc_transform.Data.obb import OBB
from noc_transform.Method.transform import transPoints
from noc_transform.Module.transform_generator import TransformGenerator
from points_shape_detect.Method.trans import normalizePointArray
from scan2cad_dataset_manage.Module.dataset_loader import DatasetLoader
from scan2cad_dataset_manage.Module.object_model_map_manager import \
    ObjectModelMapManager

from global_to_patch_retrieval.Method.feature import getPointsFeature
from global_to_patch_retrieval.Method.retrieval import getObjectRetrievalResult
from global_to_patch_retrieval.Module.retrieval_manager import RetrievalManager


class S2CRetrievalManager(RetrievalManager):
    def __init__(self,
                 scan2cad_dataset_folder_path,
                 scannet_dataset_folder_path,
                 shapenet_dataset_folder_path,
                 scannet_object_dataset_folder_path,
                 scan2cad_object_model_map_dataset_folder_path,
                 shapenet_feature_folder_path,
                 print_progress=False):
        super().__init__(shapenet_dataset_folder_path, print_progress)
        self.dataset_loader = DatasetLoader(scan2cad_dataset_folder_path,
                                            scannet_dataset_folder_path,
                                            shapenet_dataset_folder_path)
        self.object_model_map_manager = ObjectModelMapManager(
            scannet_object_dataset_folder_path, shapenet_dataset_folder_path,
            scan2cad_object_model_map_dataset_folder_path)
        self.transform_generator = TransformGenerator()
        self.uniform_feature_dict = None

        self.loadUniformFeature(shapenet_feature_folder_path)

        self.retrieval_num = 6
        return

    def loadUniformFeature(self, shapenet_feature_folder_path):
        assert os.path.exists(shapenet_feature_folder_path)

        uniform_feature_file_path = shapenet_feature_folder_path + \
            "../uniform_feature/uniform_feature.pkl"
        if not os.path.exists(uniform_feature_file_path):
            self.generateUniformFeatureDict(shapenet_feature_folder_path,
                                            uniform_feature_file_path,
                                            print_progress)

        assert os.path.exists(uniform_feature_file_path)
        with open(uniform_feature_file_path, 'rb') as f:
            self.uniform_feature_dict = pickle.load(f)
        return True

    def generateSceneRetrievalResultByNOC(self,
                                          scannet_scene_name,
                                          print_progress=False):
        render = False

        cad_file_path_list = self.uniform_feature_dict[
            'shapenet_model_file_path_list']
        cad_feature_array = self.uniform_feature_dict['feature_array']
        cad_mask_array = self.uniform_feature_dict['mask_array']

        object_filename_list = self.object_model_map_manager.getObjectFileNameList(
            scannet_scene_name)

        for object_filename in object_filename_list:
            shapenet_model_dict = self.object_model_map_manager.getShapeNetModelDict(
                scannet_scene_name, object_filename)

            scannet_object_file_path = shapenet_model_dict[
                'scannet_object_file_path']
            shapenet_model_file_path = shapenet_model_dict[
                'shapenet_model_file_path']
            trans_matrix = np.array(shapenet_model_dict['trans_matrix'])

            object_pcd = o3d.io.read_point_cloud(scannet_object_file_path)
            cad_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)

            min_point = cad_mesh.get_min_bound()
            max_point = cad_mesh.get_max_bound()

            obb = OBB.fromABBPoints(min_point, max_point)
            points = obb.points

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.transform(trans_matrix)
            obb.points = np.array(pcd.points)

            noc_trans_matrix = self.transform_generator.getNOCTransform(obb)

            object_points = np.array(object_pcd.points)

            trans_points = transPoints(object_points, noc_trans_matrix)
            object_feature, object_mask = getPointsFeature(trans_points, False)

            cad_model_file_path_list = getObjectRetrievalResult(
                object_feature, object_mask, cad_feature_array, cad_mask_array,
                cad_file_path_list, self.retrieval_num, print_progress)

            if render:
                for cad_model_file_path in cad_model_file_path_list:
                    object_pcd.points = o3d.utility.Vector3dVector(
                        trans_points)
                    retrieval_mesh = o3d.io.read_triangle_mesh(
                        cad_model_file_path)
                    points = np.array(retrieval_mesh.vertices)
                    points = normalizePointArray(points)
                    retrieval_mesh.vertices = o3d.utility.Vector3dVector(
                        points)
                    retrieval_mesh.translate([1, 0, 0])

                    points = np.array(cad_mesh.vertices)
                    points = normalizePointArray(points)
                    cad_mesh.vertices = o3d.utility.Vector3dVector(points)
                    cad_mesh.translate([2, 0, 0])

                    o3d.visualization.draw_geometries(
                        [object_pcd, retrieval_mesh, cad_mesh])
        return True

    def generateSceneRetrievalResultByLabel(self,
                                            scannet_scene_name,
                                            print_progress=False):
        render = False
        save_folder_path = "./output/retrieval/" + scannet_scene_name + "/"
        os.makedirs(save_folder_path, exist_ok=True)

        cad_file_path_list = self.uniform_feature_dict[
            'shapenet_model_file_path_list']
        cad_feature_array = self.uniform_feature_dict['feature_array']
        cad_mask_array = self.uniform_feature_dict['mask_array']

        object_filename_list = self.object_model_map_manager.getObjectFileNameList(
            scannet_scene_name)

        object_num_str = str(len(object_filename_list))

        for i, object_filename in enumerate(object_filename_list):
            if print_progress:
                print(
                    "[INFO][S2CRetrievalManager::generateSceneRetrievalResultByLabel]"
                )
                print("\t start generate retrieval results " + str(i + 1) +
                      "/" + object_num_str + "...")
            shapenet_model_dict = self.object_model_map_manager.getShapeNetModelDict(
                scannet_scene_name, object_filename)

            scannet_object_file_path = shapenet_model_dict[
                'scannet_object_file_path']
            shapenet_model_file_path = shapenet_model_dict[
                'shapenet_model_file_path']
            trans_matrix = np.array(shapenet_model_dict['trans_matrix'])
            trans_matrix_inv = np.array(
                shapenet_model_dict['trans_matrix_inv'])

            object_pcd = o3d.io.read_point_cloud(scannet_object_file_path)
            cad_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)

            min_point = cad_mesh.get_min_bound()
            max_point = cad_mesh.get_max_bound()

            obb = OBB.fromABBPoints(min_point, max_point)
            points = obb.points

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.transform(trans_matrix)
            obb.points = np.array(pcd.points)

            noc_trans_matrix = self.transform_generator.getNOCTransform(obb)
            retrieval_trans_matrix = np.linalg.inv(noc_trans_matrix)

            object_pcd.transform(trans_matrix_inv)

            object_points = np.array(object_pcd.points)
            cad_points = np.array(cad_mesh.vertices)

            cad_points, object_points = normalizePointArray(
                cad_points, object_points)

            object_feature, object_mask = getPointsFeature(
                object_points, False)

            cad_model_file_path_list = getObjectRetrievalResult(
                object_feature, object_mask, cad_feature_array, cad_mask_array,
                cad_file_path_list, self.retrieval_num, print_progress)

            if render:
                for cad_model_file_path in cad_model_file_path_list:
                    retrieval_mesh = o3d.io.read_triangle_mesh(
                        cad_model_file_path)
                    points = np.array(retrieval_mesh.vertices)
                    points = normalizePointArray(points)
                    points = transPoints(points, retrieval_trans_matrix)
                    retrieval_mesh.vertices = o3d.utility.Vector3dVector(
                        points)
                    retrieval_mesh.translate([1, 0, 0])

                    points = np.array(cad_mesh.vertices)
                    points = normalizePointArray(points)
                    cad_mesh.vertices = o3d.utility.Vector3dVector(points)
                    cad_mesh.translate([2, 0, 0])

                    o3d.visualization.draw_geometries(
                        [object_pcd, retrieval_mesh, cad_mesh])

            for j, cad_model_file_path in enumerate(cad_model_file_path_list):
                current_save_folder_path = save_folder_path + str(j) + '/'
                os.makedirs(current_save_folder_path, exist_ok=True)

                retrieval_mesh = o3d.io.read_triangle_mesh(cad_model_file_path)
                points = np.array(retrieval_mesh.vertices)
                points = normalizePointArray(points)
                points = transPoints(points, retrieval_trans_matrix)
                retrieval_mesh.vertices = o3d.utility.Vector3dVector(points)

                retrieval_mesh.compute_triangle_normals()

                save_file_path = current_save_folder_path + str(i) + "_" + str(
                    j) + ".ply"
                o3d.io.write_triangle_mesh(save_file_path, retrieval_mesh)
        return True

    def generateSceneRetrievalResult(self,
                                     scannet_scene_name,
                                     print_progress=False):
        mode_list = ['noc', 'label']
        mode = 'label'

        assert mode in mode_list

        if mode == 'noc':
            return self.generateSceneRetrievalResultByNOC(
                scannet_scene_name, print_progress)
        elif mode == 'label':
            return self.generateSceneRetrievalResultByLabel(
                scannet_scene_name, print_progress)
