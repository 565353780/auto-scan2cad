#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.kdtree import KDTree

from mesh_manage.Data.point import Point
from mesh_manage.Data.bbox import BBox
from mesh_manage.Data.channel_point import ChannelPoint

from mesh_manage.Method.io import loadFileData
from mesh_manage.Method.trans import transFormat
from mesh_manage.Method.pool import getChannelPointListWithPool


class ChannelPointCloud(object):

    def __init__(self,
                 pointcloud_file_path=None,
                 save_ignore_channel_name_list=[]):
        self.save_ignore_channel_name_list = save_ignore_channel_name_list

        self.channel_point_list = []
        self.kd_tree = None
        self.xyz_changed = True

        if pointcloud_file_path is not None:
            self.loadData(pointcloud_file_path)
        return

    def reset(self):
        self.channel_point_list.clear()
        self.kd_tree = None
        self.xyz_changed = True
        return True

    def loadData(self, pointcloud_file_path, print_progress=False):
        self.reset()

        if print_progress:
            print("[INFO][ChannelPointCloud::loadData]")
            print("\t start load pointcloud :")
            print("\t pointcloud_file_path =", pointcloud_file_path)

        channel_name_list, channel_value_list_list, _ = loadFileData(
            pointcloud_file_path, True)

        assert channel_name_list != [] or channel_value_list_list != []

        channel_point_list = getChannelPointListWithPool(
            channel_name_list, channel_value_list_list)
        self.channel_point_list.extend(channel_point_list)

        self.updateKDTree()
        return True

    def getChannelNameList(self):
        if len(self.channel_point_list) == 0:
            return []
        return self.channel_point_list[0].getChannelNameList()

    def getChannelValueList(self, channel_name):
        channel_value_list = []
        for channel_point in self.channel_point_list:
            channel_value_list.append(
                channel_point.getChannelValue(channel_name))
        return channel_value_list

    def getChannelValueListList(self, channel_name_list):
        channel_value_list_list = []
        for channel_point in self.channel_point_list:
            channel_value_list = []
            for channel_name in channel_name_list:
                channel_value_list.append(
                    channel_point.getChannelValue(channel_name))
            channel_value_list_list.append(channel_value_list)
        return channel_value_list_list

    def getBBox(self):
        xyz_list = self.getChannelValueListList(["x", "y", "z"])
        assert len(xyz_list) > 0
        assert None not in xyz_list[0]

        xyz_array = np.array(xyz_list)
        bbox = BBox(
            Point(min(xyz_array[:, 0]), min(xyz_array[:, 1]),
                  min(xyz_array[:, 2])),
            Point(max(xyz_array[:, 0]), max(xyz_array[:, 1]),
                  max(xyz_array[:, 2])))
        return bbox

    def updateKDTree(self):
        if not self.xyz_changed:
            return True

        self.kd_tree = None
        xyz_list = self.getChannelValueListList(["x", "y", "z"])
        if len(xyz_list) == 0:
            return False
        if None in xyz_list[0]:
            return False
        self.kd_tree = KDTree(xyz_list)
        self.xyz_changed = False
        return True

    def addChannelPoint(self, channel_name_list, channel_value_list):
        channel_point = ChannelPoint(channel_name_list, channel_value_list)
        self.channel_point_list.append(channel_point)
        return True

    def addChannelPointList(self,
                            channel_name_list,
                            channel_value_list_list,
                            print_progress=False):
        if len(channel_name_list) == 0:
            print("[WARN][ChannelPointCloud::addChannelPointList]")
            print("\t channel_name_list is empty!")
            return True

        if len(channel_value_list_list) == 0:
            print("[WARN][ChannelPointCloud::addChannelPointList]")
            print("\t channel_value_list_list is empty!")
            return True

        if print_progress:
            print("[INFO][ChannelPointCloud::addChannelPointList]")
            print("\t start add channel point list...")

        channel_point_list = getChannelPointListWithPool(
            channel_name_list, channel_value_list_list)
        self.channel_point_list.extend(channel_point_list)
        return True

    def getChannelPoint(self, channel_point_idx):
        assert channel_point_idx < len(self.channel_point_list)
        return self.channel_point_list[channel_point_idx]

    def getFilterChannelPointCloud(self, point_idx_list):
        channel_name_list = self.getChannelNameList()
        channel_pointcloud = ChannelPointCloud()

        for channel_point_idx in point_idx_list:
            channel_point = self.getChannelPoint(channel_point_idx)
            assert channel_point is not None

            channel_value_list = channel_point.getChannelValueList(
                channel_name_list)
            channel_pointcloud.addChannelPoint(channel_name_list,
                                               channel_value_list)
        return channel_pointcloud

    def getNearestPointInfo(self, x, y, z):
        self.updateKDTree()

        if self.kd_tree is None:
            return None, None
        if len(self.channel_point_list) == 0:
            return None, None
        nearest_dist, nearest_point_idx = self.kd_tree.query([x, y, z])
        return nearest_dist, nearest_point_idx

    def getNearestDist(self, x, y, z):
        self.updateKDTree()

        if self.kd_tree is None:
            return None
        if len(self.channel_point_list) == 0:
            return None
        nearest_dist, _ = self.kd_tree.query([x, y, z])
        return nearest_dist

    def getSelfNearestDist(self, channel_point_idx):
        self.updateKDTree()

        if self.kd_tree is None:
            return None
        if len(self.channel_point_list) == 0:
            return None
        if channel_point_idx >= len(self.channel_point_list):
            return None
        xyz = self.channel_point_list[channel_point_idx].getChannelValueList(
            ["x", "y", "z"])
        if None in xyz:
            return None
        nearest_dist_list, _ = self.kd_tree.query([xyz[0], xyz[1], xyz[2]], 2)
        return nearest_dist_list[1]

    def getNearestPoint(self, x, y, z):
        _, nearest_point_idx = self.getNearestPointInfo(x, y, z)
        if nearest_point_idx is None:
            return None
        return self.channel_point_list[nearest_point_idx]

    def getNearestChannelValueListValue(self, x, y, z, channel_name_list):
        nearest_channel_value_list = []
        if len(channel_name_list) == 0:
            return nearest_channel_value_list
        nearest_point = self.getNearestPoint(x, y, z)
        if nearest_point is None:
            return None
        for channel_name in channel_name_list:
            nearest_channel_value_list.append(
                nearest_point.getChannelValue(channel_name))
        return nearest_channel_value_list

    def copyChannelValue(self,
                         target_pointcloud,
                         channel_name_list,
                         print_progress=False):
        pointcloud_size = len(self.channel_point_list)
        target_pointcloud_size = len(target_pointcloud.channel_point_list)
        assert target_pointcloud_size > 0

        assert pointcloud_size == target_pointcloud_size

        first_point_channel_value_list = \
            target_pointcloud.channel_point_list[0].getChannelValueList(channel_name_list)
        assert None not in first_point_channel_value_list

        print("[INFO][ChannelPointCloud::copyChannelValue]")
        print("\t start copy channel value :")
        print("\t channel_name_list = [", end="")
        for channel_name in channel_name_list:
            print(" " + channel_name, end="")
        print(" ]...")
        channel_value_list_list = \
            target_pointcloud.getChannelValueListList(channel_name_list)

        if pointcloud_size == 0:
            if print_progress:
                for channel_value_list in tqdm(channel_value_list_list):
                    new_point = ChannelPoint()
                    new_point.setChannelValueList(channel_name_list,
                                                  channel_value_list)
                    self.channel_point_list.append(new_point)
            else:
                for channel_value_list in channel_value_list_list:
                    new_point = ChannelPoint()
                    new_point.setChannelValueList(channel_name_list,
                                                  channel_value_list)
                    self.channel_point_list.append(new_point)

            self.updateKDTree()
            return True

        if print_progress:
            for i in tqdm(range(pointcloud_size)):
                channel_value_list = channel_value_list_list[i]
                self.channel_point_list[i].setChannelValueList(
                    channel_name_list, channel_value_list)
        else:
            for i in range(pointcloud_size):
                channel_value_list = channel_value_list_list[i]
                self.channel_point_list[i].setChannelValueList(
                    channel_name_list, channel_value_list)

        self.updateKDTree()
        return True

    def setChannelValueByKDTree(self,
                                target_pointcloud,
                                channel_name_list,
                                print_progress=False):
        self.updateKDTree()

        if len(self.channel_point_list) == 0:
            return True

        assert len(target_pointcloud.channel_point_list) > 0

        first_point_xyz = self.channel_point_list[0].getChannelValueList(
            ["x", "y", "z"])
        assert None not in first_point_xyz

        first_point_channel_value_list = \
            target_pointcloud.getNearestChannelValueListValue(
                first_point_xyz[0],
                first_point_xyz[1],
                first_point_xyz[2],
                channel_name_list)
        assert None not in first_point_channel_value_list

        if print_progress:
            print("[INFO][ChannelPointCloud::setChannelValueByKDTree]")
            print("\t start set channel value by KDTree :")
            print("\t channel_name_list = [", end="")
            for channel_name in channel_name_list:
                print(" " + channel_name, end="")
            print(" ]...")
            for channel_point in tqdm(self.channel_point_list):
                xyz = channel_point.getChannelValueList(["x", "y", "z"])
                channel_value_list = \
                    target_pointcloud.getNearestChannelValueListValue(xyz[0],
                                                                      xyz[1],
                                                                      xyz[2],
                                                                      channel_name_list)
                channel_point.setChannelValueList(channel_name_list,
                                                  channel_value_list)
        else:
            for channel_point in self.channel_point_list:
                xyz = channel_point.getChannelValueList(["x", "y", "z"])
                channel_value_list = \
                    target_pointcloud.getNearestChannelValueListValue(xyz[0],
                                                                      xyz[1],
                                                                      xyz[2],
                                                                      channel_name_list)
                channel_point.setChannelValueList(channel_name_list,
                                                  channel_value_list)
        return True

    def removeOutlierPoints(self, outlier_dist_max, print_progress=False):
        self.updateKDTree()

        assert self.kd_tree is not None

        assert outlier_dist_max > 0

        remove_point_idx_list = []

        for_data = range(len(self.channel_point_list))
        if print_progress:
            print("[INFO][ChannelPointCloud::removeOutlierPoints]")
            print("\t start remove outerlier points with outlier_dist_max = " +
                  str(outlier_dist_max) + "...")
            for_data = tqdm(for_data)
        for i in for_data:
            current_nearest_dist = self.getSelfNearestDist(i)
            if current_nearest_dist > outlier_dist_max:
                remove_point_idx_list.append(i)

        if len(remove_point_idx_list) == 0:
            return True

        for i in range(len(remove_point_idx_list)):
            self.channel_point_list.pop(remove_point_idx_list[i] - i)

        if print_progress:
            print("[INFO][ChannelPointCloud::removeOutlierPoints]")
            print("\t removed " + str(len(remove_point_idx_list)) +
                  " points...")
        return True

    def paintByLabel(self,
                     label_channel_name,
                     color_map,
                     print_progress=False):
        if len(self.channel_point_list) == 0:
            return True

        first_point_label_value = self.channel_point_list[0].getChannelValue(
            label_channel_name)

        assert first_point_label_value is not None

        color_map_size = len(color_map)

        assert color_map_size > 0

        if print_progress:
            print("[INFO][ChannelPointCloud::paintByLabel]")
            print("\t start paint by label...")
            for channel_point in tqdm(self.channel_point_list):
                label_value = channel_point.getChannelValue(label_channel_name)
                rgb = color_map[label_value % color_map_size]
                channel_point.setChannelValueList(["r", "g", "b"], rgb)
        else:
            for channel_point in self.channel_point_list:
                label_value = channel_point.getChannelValue(label_channel_name)
                rgb = color_map[label_value % color_map_size]
                channel_point.setChannelValueList(["r", "g", "b"], rgb)
        return True

    def getPCDHeader(self):
        channel_list = []

        point_num = len(self.channel_point_list)
        if point_num > 0:
            channel_list = self.channel_point_list[0].channel_list

        pcd_header = "# .PCD v0.7 - Point Cloud Data file format\n"
        pcd_header += "VERSION 0.7\n"

        pcd_header += "FIELDS"
        for channel in channel_list:
            if channel.name in self.save_ignore_channel_name_list:
                continue
            pcd_header += " " + channel.name
        pcd_header += "\n"

        pcd_header += "SIZE"
        for channel in channel_list:
            if channel.name in self.save_ignore_channel_name_list:
                continue
            pcd_header += " " + str(channel.size)
        pcd_header += "\n"

        pcd_header += "TYPE"
        for channel in channel_list:
            if channel.name in self.save_ignore_channel_name_list:
                continue
            pcd_header += " " + channel.type
        pcd_header += "\n"

        pcd_header += "COUNT"
        for channel in channel_list:
            if channel.name in self.save_ignore_channel_name_list:
                continue
            pcd_header += " " + str(channel.count)
        pcd_header += "\n"

        pcd_header += "WIDTH " + str(point_num) + "\n"

        pcd_header += "HEIGHT 1\n"
        pcd_header += "VIEWPOINT 0 0 0 1 0 0 0\n"

        pcd_header += "POINTS " + str(point_num) + "\n"

        pcd_header += "DATA ascii\n"
        return pcd_header

    def savePCD(self, save_pointcloud_file_path, print_progress=False):
        with open(save_pointcloud_file_path, "w") as f:
            pcd_header = self.getPCDHeader()
            f.write(pcd_header)

            for_data = self.channel_point_list
            if print_progress:
                print("[INFO][ChannelPointCloud::savePCD]")
                print("\t start save pointcloud to" +
                      save_pointcloud_file_path + "...")
                for_data = tqdm(self.channel_point_list)
            for channel_point in for_data:
                last_channel_idx = len(channel_point.channel_list) - 1
                for i in range(last_channel_idx + 1):
                    if channel_point.channel_list[
                            i].name in self.save_ignore_channel_name_list:
                        if i == last_channel_idx:
                            f.write("\n")
                        continue
                    f.write(str(channel_point.channel_list[i].value))
                    if i < last_channel_idx:
                        f.write(" ")
                    else:
                        f.write("\n")
        return True

    def savePointCloud(self, save_pointcloud_file_path, print_progress=False):
        if save_pointcloud_file_path[:-4] == ".pcd":
            assert self.savePCD(save_pointcloud_file_path, print_progress)

        if save_pointcloud_file_path[:-4] == ".ply":
            pcd_pointcloud_file_path = save_pointcloud_file_path[:-4] + ".pcd"
            pcd_save_idx = 0
            while os.path.exists(pcd_pointcloud_file_path):
                pcd_pointcloud_file_path = save_pointcloud_file_path[:-4] + "_" + str(
                    pcd_save_idx) + ".pcd"
                pcd_save_idx += 1

            assert self.savePCD(pcd_pointcloud_file_path, print_progress)
            assert transFormat(pcd_pointcloud_file_path,
                               save_pointcloud_file_path, True, print_progress)
        return True

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[ChannelPointCloud]")
        print(line_start + "\t channel_point num =",
              len(self.channel_point_list))
        if len(self.channel_point_list) > 0:
            print(line_start + "\t channel_point channel =")
            self.channel_point_list[0].outputInfo(info_level + 1)
        return True


def demo():
    pointcloud_file_path = \
        "/home/chli/chLi/OBJs/OpenGL/bunny_1.ply"

    channel_pointcloud = ChannelPointCloud()
    channel_pointcloud.loadData(pointcloud_file_path)
    channel_pointcloud.outputInfo()
    return True
