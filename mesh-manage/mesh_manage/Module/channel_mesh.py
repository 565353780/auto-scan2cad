#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mesh_manage.Data.face_set import FaceSet

from mesh_manage.Method.io import loadFileData, saveChannelMesh

from mesh_manage.Data.channel_pointcloud import ChannelPointCloud


class ChannelMesh(ChannelPointCloud):

    def __init__(self,
                 mesh_file_path=None,
                 save_ignore_channel_name_list=[],
                 load_point_only=False):
        super(ChannelMesh, self).__init__(None, save_ignore_channel_name_list)
        self.face_set = FaceSet()

        if mesh_file_path is not None:
            self.loadData(mesh_file_path, load_point_only)
        return

    @classmethod
    def fromChannelPointCloud(cls, channel_pointcloud):
        channel_mesh = cls(None,
                           channel_pointcloud.save_ignore_channel_name_list,
                           True)
        channel_mesh.channel_point_list = channel_pointcloud.channel_point_list
        channel_mesh.kd_tree = channel_pointcloud.kd_tree
        channel_mesh.xyz_changed = channel_pointcloud.xyz_changed
        return channel_mesh

    def reset(self):
        super(ChannelMesh, self).reset()
        self.face_set.reset()
        return True

    def loadData(self,
                 mesh_file_path,
                 load_point_only=False,
                 print_progress=False):
        self.reset()

        if print_progress:
            print("[INFO][ChannelMesh::loadData]")
            print("\t start load mesh :")
            print("\t mesh_file_path =", mesh_file_path)

        channel_name_list, channel_value_list_list, point_idx_list_list = \
            loadFileData(mesh_file_path, load_point_only)

        if channel_name_list == [] or channel_value_list_list == []:
            print("[ERROR][ChannelMesh::loadData]")
            print("\t loadFileData failed!")
            return False

        self.addChannelPointList(channel_name_list, channel_value_list_list,
                                 print_progress)

        self.updateKDTree()

        if point_idx_list_list != []:
            for point_idx_list in point_idx_list_list:
                self.addFace(point_idx_list)
        return True

    def addFace(self, point_idx_list):
        return self.face_set.addFace(point_idx_list)

    def addFaceSet(self, face_set, point_start_idx):
        return self.face_set.addFaceSet(face_set, point_start_idx)

    def getFace(self, face_idx):
        return self.face_set.getFace(face_idx)

    def getFaceIdxListInPointIdxList(self, point_idx_list):
        return self.face_set.getFaceIdxListInPointIdxList(point_idx_list)

    def getPointIdxListFromFaceIdxList(self, face_idx_list):
        return self.face_set.getPointIdxListAndMappingDict(face_idx_list)[0]

    def getChannelMeshByFace(self, face_idx_list):
        point_idx_list, mapping_dict = self.face_set.getPointIdxListAndMappingDict(
            face_idx_list)
        if point_idx_list is None or mapping_dict is None:
            print("[ERROR][ChannelMesh::getChannelMeshByFace]")
            print("\t getPointIdxListAndMappingDict failed!")
            return None

        channel_pointcloud = self.getFilterChannelPointCloud(point_idx_list)
        if channel_pointcloud is None:
            print("[ERROR][ChannelMesh::getChannelMeshByFace]")
            print("\t getFilterChannelPointCloud failed!")
            return None

        face_set = self.face_set.getMappingFaceSet(face_idx_list, mapping_dict)
        if face_set is None:
            print("[ERROR][ChannelMesh::getChannelMeshByFace]")
            print("\t getMappingFaceSet failed!")
            return None

        channel_mesh = ChannelMesh.fromChannelPointCloud(channel_pointcloud)
        channel_mesh.face_set = face_set
        return channel_mesh

    def getChannelMeshByPoint(self, point_idx_list):
        channel_pointcloud = self.getFilterChannelPointCloud(point_idx_list)
        if channel_pointcloud is None:
            print("[ERROR][ChannelMesh::getChannelMeshByPoint]")
            print("\t getFilterChannelPointCloud failed!")
            return None

        face_set = self.face_set.getFaceSetInPointIdxList(point_idx_list)
        if face_set is None:
            print("[ERROR][ChannelMesh::getChannelMeshByPoint]")
            print("\t getFaceSetInPointIdxList failed!")
            return None

        channel_mesh = ChannelMesh.fromChannelPointCloud(channel_pointcloud)
        channel_mesh.face_set = face_set
        return channel_mesh

    def saveMesh(self, save_file_path, print_progress=False):
        if not saveChannelMesh(self, save_file_path, print_progress):
            print("[ERROR][ChannelMesh::saveMesh]")
            print("\t saveChannelMesh failed!")
            return False
        return True

    def generateMeshByFace(self,
                           face_idx_list,
                           save_file_path,
                           print_progress=False):
        channel_mesh = self.getChannelMeshByFace(face_idx_list)
        if channel_mesh is None:
            print("[ERROR][ChannelMesh::generateMeshByFace]")
            print("\t getChannelMeshByFace failed!")
            return False
        if not channel_mesh.saveMesh(save_file_path, print_progress):
            print("[ERROR][ChannelMesh::generateMeshByFace]")
            print("\t saveMesh failed!")
            return False
        return True

    def generateMeshByPoint(self,
                            point_idx_list,
                            save_file_path,
                            print_progress=False):
        channel_mesh = self.getChannelMeshByPoint(point_idx_list)
        if channel_mesh is None:
            print("[ERROR][ChannelMesh::generateMeshByPoint]")
            print("\t getChannelMeshByPoint failed!")
            return False
        if not channel_mesh.saveMesh(save_file_path, print_progress):
            print("[ERROR][ChannelMesh::generateMeshByPoint]")
            print("\t saveMesh failed!")
            return False
        return True

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[ChannelMesh]")
        print(line_start + "\t channel_pointcloud =")
        super(ChannelMesh, self).outputInfo(info_level + 1)
        print(line_start + "\t face_set =")
        self.face_set.outputInfo(info_level + 1)
        return True


def demo():
    mesh_file_path = "/home/chli/chLi/ScanNet/scans/scene0474_02/scene0474_02_vh_clean_2.ply"

    channel_mesh = ChannelMesh(mesh_file_path)

    face_idx_list = [i for i in range(20)]

    point_idx_list = channel_mesh.getPointIdxListFromFaceIdxList(face_idx_list)

    channel_mesh.generateMeshByFace(face_idx_list,
                                    "/home/chli/chLi/channel_mesh/test1.ply")
    channel_mesh.generateMeshByPoint(point_idx_list,
                                     "/home/chli/chLi/channel_mesh/test2.ply")
    return True
