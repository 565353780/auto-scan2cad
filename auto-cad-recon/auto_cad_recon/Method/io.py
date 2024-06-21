#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os

import numpy as np
import open3d as o3d
from tqdm import tqdm
from multiprocessing import Process

from scannet_sim_manage.Method.mesh import getCameraBoundaryMesh

from auto_cad_recon.Config.color import COLOR_MAP
from auto_cad_recon.Method.path import createFileFolder
from auto_cad_recon.Method.render import getObjectRenderList, getSceneRenderList
from auto_cad_recon.Method.scene_render import getMergedScenePCD
from auto_cad_recon.Method.cad_render import getGTMeshInfoList


def writePointCloud(save_file_path, point_cloud, print_progress=False):
    process = Process(
        target=o3d.io.write_point_cloud,
        args=(save_file_path, point_cloud, True, False, print_progress),
    )
    process.start()
    #  process.join()
    #  process.close()
    return True


def toJson(data):
    json_dict = {}
    for first_key in data.keys():
        json_dict[first_key] = {}
        for key, value in data[first_key].items():
            if isinstance(value, np.ndarray):
                json_dict[first_key][key] = value.tolist()
                continue

            if isinstance(value, str) or isinstance(value, list):
                json_dict[first_key][key] = value
    return json_dict


def saveData(data, save_json_file_path):
    createFileFolder(save_json_file_path)

    json_dict = toJson(data)

    with open(save_json_file_path, "w") as f:
        json.dump(json_dict, f, indent=4)
    return True


def saveDataList(data_list, save_json_file_path):
    save_dict = {}
    for i in range(len(data_list)):
        save_dict[str(i)] = toJson(data_list[i])

    createFileFolder(save_json_file_path)

    with open(save_json_file_path, "w") as f:
        json.dump(save_dict, f)
    return True


def loadData(json_file_path):
    assert os.path.exists(json_file_path)

    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data


def loadDataList(json_file_path):
    assert os.path.exists(json_file_path)

    with open(json_file_path, "r") as f:
        data = json.load(f)
    return list(data.values())


def saveRenderResult(save_folder_path, render_list, label_list, print_progress=False):
    assert len(render_list) == len(label_list)

    for_data = range(len(render_list))
    if print_progress:
        #  print("[INFO][io::saveRenderResult]")
        #  print("\t start saving render results...")
        for_data = tqdm(for_data)
    for i in for_data:
        geometry = render_list[i]
        label = label_list[i]

        save_file_path = save_folder_path + label
        createFileFolder(save_file_path)

        file_type = label.split(".")[-1]

        assert file_type in ["pcd", "ply", "lines"]

        if file_type == "pcd":
            save_file_path = save_file_path.replace(".pcd", ".ply")
            o3d.io.write_point_cloud(save_file_path, geometry, write_ascii=True)
            continue
        if file_type == "ply":
            o3d.io.write_triangle_mesh(save_file_path, geometry, write_ascii=True)
            continue
        if file_type == "lines":
            save_file_path = save_file_path.replace(".lines", ".ply")
            o3d.io.write_line_set(save_file_path, geometry, write_ascii=True)
            continue
    return True


def saveCameraBoundarys(point_image_list, save_folder_path):
    if len(point_image_list) == 0:
        return True

    os.makedirs(save_folder_path, exist_ok=True)

    for i, point_image in enumerate(point_image_list):
        camera_boundary_mesh = getCameraBoundaryMesh(
            point_image.boundary_point_array, point_image.camera_point
        )

        save_file_path = save_folder_path + str(i) + ".ply"
        o3d.io.write_triangle_mesh(
            save_file_path, camera_boundary_mesh, write_ascii=True
        )
    return True


def saveGTMeshInfo(save_folder_path, data_list, translate=[0, 0, 0]):
    os.makedirs(save_folder_path, exist_ok=True)

    mesh_info_list = getGTMeshInfoList(data_list, translate)

    for i, mesh_info in enumerate(mesh_info_list):
        mesh_file_path, trans_matrix, obb_pcd = mesh_info
        mesh_info_dict = {
            "mesh_file_path": mesh_file_path,
            "trans_matrix": trans_matrix,
            "obb": np.array(obb_pcd.points),
        }

        mesh_info_npy = save_folder_path + str(i) + ".npy"

        np.save(mesh_info_npy, mesh_info_dict)
    return True


def saveAllRenderResult(
    save_folder_path,
    data_list,
    layout_data=None,
    point_image_list=None,
    object_mode="none",
    obb_mode="none",
    mesh_mode="none",
    object_color_map=COLOR_MAP,
    object_translate=[0, 0, 0],
    obb_color_map=COLOR_MAP,
    obb_translate=[0, 0, 0],
    mesh_color_map=COLOR_MAP,
    mesh_translate=[0, 0, 0],
    scene_background_only=True,
    is_scene_gray=True,
    scene_translate=[0, 0, 0],
    estimate_scene_normals=False,
    print_progress=False,
):
    object_render_list, _, object_label_list = getObjectRenderList(
        data_list,
        object_mode,
        obb_mode,
        mesh_mode,
        object_color_map,
        object_translate,
        obb_color_map,
        obb_translate,
        mesh_color_map,
        mesh_translate,
    )

    scene_render_list, _, scene_label_list = getSceneRenderList(
        layout_data,
        point_image_list,
        scene_background_only,
        is_scene_gray,
        scene_translate,
        estimate_scene_normals,
    )

    render_list = object_render_list + scene_render_list
    label_list = object_label_list + scene_label_list

    save_object_folder_path = save_folder_path + "objects/"
    save_scene_folder_path = save_folder_path + "scene/"

    if print_progress:
        print("[INFO][io::saveAllRenderResult]")
        print("\t start saving object render results...")
    saveRenderResult(
        save_object_folder_path, object_render_list, object_label_list, print_progress
    )
    if print_progress:
        print("[INFO][io::saveAllRenderResult]")
        print("\t start saving scene render results...")
    saveRenderResult(
        save_scene_folder_path, scene_render_list, scene_label_list, print_progress
    )

    if point_image_list is not None:
        scene_background_only = False
        is_scene_gray = False
        estimate_scene_normals = True
        merged_scene_pcd = getMergedScenePCD(
            point_image_list,
            scene_background_only,
            is_scene_gray,
            scene_translate,
            estimate_scene_normals,
        )
        save_scene_merge_pcd_file_path = save_scene_folder_path + "scene_merge_pcd.ply"
        return True

        o3d.io.write_point_cloud(
            save_scene_merge_pcd_file_path,
            merged_scene_pcd,
            write_ascii=True,
            print_progress=print_progress,
        )

        save_camera_boundary_folder_path = save_folder_path + "camera_boundary/"
        saveCameraBoundarys(point_image_list, save_camera_boundary_folder_path)
    return True
