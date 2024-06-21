#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from mesh_manage.Config.move import MOVE_LIST_DICT

from mesh_manage.Method.heatmap import getHeatMap, getDiffHeatMap


def demo_heatmap():
    coscan_partial_mesh_file_path = \
        "/home/chli/chLi/coscan_data/scene_result/matterport3d_03/coscan/scene_19.ply"
    complete_mesh_file_path = \
        "/home/chli/chLi/coscan_data/scene_result/matterport3d_03/matterport_03_cut.ply"
    coscan_save_complete_mesh_file_path = \
        "/home/chli/chLi/coscan_data/scene_result/matterport3d_03/comp_coscan.ply"
    move_list = MOVE_LIST_DICT["matterport3d_03"]
    error_max = 0.5
    is_visual = False
    print_progress = True

    print("===========================")
    print("==== start get heatmap ====")
    print("===========================")
    getHeatMap(coscan_partial_mesh_file_path,
               complete_mesh_file_path,
               coscan_save_complete_mesh_file_path,
               move_list=move_list,
               error_max=error_max,
               is_visual=is_visual,
               print_progress=print_progress)
    return True


def demo_ff_heatmap():
    for idx in range(28):
    #  for idx in [0, 27]:
        partial_mesh_file_path = \
            "/home/chli/chLi/coscan_data/fast_forward/ff_recon_result/merged_vpp_mesh_" + str(idx) + ".ply"
        complete_mesh_file_path = \
            "/home/chli/chLi/coscan_data/fast_forward/company_source_remesh_cut.ply"
        save_complete_mesh_file_path = \
            "/home/chli/chLi/coscan_data/fast_forward/ff_recon_result_render_gray_blue/merged_vpp_mesh_" + str(idx) + ".ply"
        move_list = MOVE_LIST_DICT["company"]
        error_max = 0.5
        use_icp = False
        is_visual = False
        print_progress = True

        if os.path.exists(save_complete_mesh_file_path):
            continue

        tmp_save_complete_mesh_file_path = save_complete_mesh_file_path[:-4] + "_tmp.ply"

        print("===========================")
        print("==== start get heatmap ====")
        print("===========================")
        print(idx + 1, "/", 28)
        getHeatMap(partial_mesh_file_path,
                   complete_mesh_file_path,
                   tmp_save_complete_mesh_file_path,
                   move_list=move_list,
                   error_max=error_max,
                   use_icp=use_icp,
                   is_visual=is_visual,
                   print_progress=print_progress)

        os.rename(tmp_save_complete_mesh_file_path,
                  save_complete_mesh_file_path)
    return


def demo_diffheatmap():
    coscan_partial_mesh_file_path = \
        "/home/chli/chLi/coscan_data/scene_result/matterport3d_03/coscan/scene_19.ply"
    dong_partial_mesh_file_path = \
        "/home/chli/chLi/coscan_data/scene_result/matterport3d_03/dong/scene_21.ply"
    complete_mesh_file_path = \
        "/home/chli/chLi/coscan_data/scene_result/matterport3d_03/matterport_03_cut.ply"
    save_complete_mesh_file_path = \
        "/home/chli/chLi/coscan_data/scene_result/matterport3d_03/diff_heatmap.ply"
    move_list = MOVE_LIST_DICT["matterport3d_03"]
    error_max = 0.5
    is_visual = False
    print_progress = True

    print("================================")
    print("==== start get diff heatmap ====")
    print("================================")
    getDiffHeatMap(coscan_partial_mesh_file_path,
                   dong_partial_mesh_file_path,
                   complete_mesh_file_path,
                   save_complete_mesh_file_path,
                   move_list=move_list,
                   error_max=error_max,
                   is_visual=is_visual,
                   print_progress=print_progress)
    return True


def demo_coscan():
    scene_result_folder_path = "/home/chli/chLi/coscan_data/scene_result/"
    #  scene_name = "front3d_19"
    #  scene_name = "matterport3d_01"
    #  scene_name = "matterport3d_03"
    scene_name = "matterport3d_05"
    move_list = MOVE_LIST_DICT[scene_name]
    error_max = 0.5
    is_visual = False
    print_progress = True

    work_dict_collection = {
        "front3d_19": {
            "coscan_partial_mesh_file_path":
            scene_result_folder_path + "front3d_19/coscan/scene_29.ply",
            "dong_partial_mesh_file_path":
            scene_result_folder_path + "front3d_19/dong/scene_27.ply",
            "complete_mesh_file_path":
            scene_result_folder_path + "front3d_19/19_cut.ply",
        },
        "matterport3d_01": {
            "coscan_partial_mesh_file_path":
            scene_result_folder_path + "matterport3d_01/coscan/scene_29.ply",
            "dong_partial_mesh_file_path":
            scene_result_folder_path + "matterport3d_01/dong/scene_24.ply",
            "complete_mesh_file_path":
            scene_result_folder_path + "matterport3d_01/matterport_01_cut.ply",
        },
        "matterport3d_03": {
            "coscan_partial_mesh_file_path":
            scene_result_folder_path + "matterport3d_03/coscan/scene_19.ply",
            "dong_partial_mesh_file_path":
            scene_result_folder_path + "matterport3d_03/dong/scene_21.ply",
            "complete_mesh_file_path":
            scene_result_folder_path + "matterport3d_03/matterport_03_cut.ply",
        },
        "matterport3d_05": {
            "coscan_partial_mesh_file_path":
            scene_result_folder_path + "matterport3d_05/coscan/scene_33.ply",
            "dong_partial_mesh_file_path":
            scene_result_folder_path + "matterport3d_05/dong/scene_16.ply",
            "complete_mesh_file_path":
            scene_result_folder_path + "matterport3d_05/matterport_05_cut.ply",
        },
    }

    # Auto generate, no need to edit
    work_dict = work_dict_collection[scene_name]
    coscan_partial_mesh_file_path = work_dict["coscan_partial_mesh_file_path"]
    dong_partial_mesh_file_path = work_dict["dong_partial_mesh_file_path"]
    complete_mesh_file_path = work_dict["complete_mesh_file_path"]
    coscan_save_complete_mesh_file_path = scene_result_folder_path + \
        scene_name + "/comp_coscan.ply"
    dong_save_complete_mesh_file_path = scene_result_folder_path + \
        scene_name + "/comp_dong.ply"
    save_complete_mesh_file_path = scene_result_folder_path + \
        scene_name + "/diff_heatmap.ply"

    print("================================")
    print("==== start get diff heatmap ====")
    print("================================")
    getDiffHeatMap(coscan_partial_mesh_file_path,
                   dong_partial_mesh_file_path,
                   complete_mesh_file_path,
                   save_complete_mesh_file_path,
                   move_list=move_list,
                   error_max=error_max,
                   is_visual=is_visual,
                   print_progress=print_progress)
    return

    print("==================================")
    print("==== start get coscan heatmap ====")
    print("==================================")
    getHeatMap(coscan_partial_mesh_file_path,
               complete_mesh_file_path,
               coscan_save_complete_mesh_file_path,
               move_list=move_list,
               error_max=error_max,
               is_visual=is_visual,
               print_progress=print_progress)

    print("================================")
    print("==== start get dong heatmap ====")
    print("================================")
    getHeatMap(dong_partial_mesh_file_path,
               complete_mesh_file_path,
               dong_save_complete_mesh_file_path,
               move_list=move_list,
               error_max=error_max,
               is_visual=is_visual,
               print_progress=print_progress)
    return True


if __name__ == "__main__":
    #  demo_heatmap()
    demo_ff_heatmap()
    #  demo_diffheatmap()
    #  demo_coscan()
