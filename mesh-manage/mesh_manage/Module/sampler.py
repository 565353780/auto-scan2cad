#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mesh_manage.Method.sample import samplePointCloud, sampleMesh

class Sampler(object):
    def __init__(self):
        return

    def samplePointCloud(self,
                         pointcloud_file_path,
                         down_sample_cluster_num,
                         save_pointcloud_file_path,
                         print_progress=False):
        if not samplePointCloud(pointcloud_file_path,
                                down_sample_cluster_num,
                                save_pointcloud_file_path,
                                print_progress):
            print("[ERROR][Sampler::samplePointCloud]")
            print("\t samplePointCloud failed!")
            return False
        return True

    def sampleMesh(self,
                   mesh_file_path,
                   vertex_cluster_dist_max,
                   save_mesh_file_path,
                   print_progress=False):
        if not sampleMesh(mesh_file_path,
                          vertex_cluster_dist_max,
                          save_mesh_file_path,
                          print_progress):
            print("[ERROR][Sampler::sampleMesh]")
            print("\t sampleMesh failed!")
            return False
        return True

def demo_sample_pointcloud():
    pointcloud_file_path = "/home/chli/chLi/OBJs/OpenGL/bunny_1.pcd"
    down_sample_cluster_num = 8
    save_pointcloud_file_path = "/home/chli/chLi/OBJs/OpenGL/bunny_1_sampled.pcd"

    sampler = Sampler()
    sampler.samplePointCloud(pointcloud_file_path,
                             down_sample_cluster_num,
                             save_pointcloud_file_path)
    return True

def demo_sample_mesh():
    mesh_file_path = "/home/chli/.gazebo/models/MatterPort/03/matterport_03_bed_source.ply"
    vertex_cluster_dist_max = 0.0001
    save_mesh_file_path = "/home/chli/.gazebo/models/MatterPort/03/matterport_03_bed_source_0001.ply"

    mesh_sampler = Sampler()
    mesh_sampler.sampleMesh(mesh_file_path,
                            vertex_cluster_dist_max,
                            save_mesh_file_path)
    return True

