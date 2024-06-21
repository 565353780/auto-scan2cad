#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mesh_manage.Method.list import isListInList

class Face(object):
    def __init__(self, point_idx_list):
        self.point_idx_list = point_idx_list
        return

    def isInPointIdxList(self, point_idx_list):
        return isListInList(self.point_idx_list, point_idx_list)

    def isSameFace(self, face):
        if len(self.point_idx_list) != len(face.point_idx_list):
            return False

        return self.isInPointIdxList(face.point_idx_list)

    def getMappingPointIdxList(self, mapping_dict):
        mapping_point_idx_list = []
        for point_idx in self.point_idx_list:
            if str(point_idx) not in mapping_dict.keys():
                print("[ERROR][Face::getMappingFace]")
                print("\t mapping_dict not valid for this face!")
                return None

            mapping_point_idx = mapping_dict[str(point_idx)]
            mapping_point_idx_list.append(mapping_point_idx)

        return mapping_point_idx_list

    def getMappingFace(self, mapping_dict):
        mapping_point_idx_list = self.getMappingPointIdxList(mapping_dict)
        if mapping_point_idx_list is None:
            print("[ERROR][Face::getMappingFace]")
            print("\t getMappingPointIdxList failed!")
            return None

        mapping_face = Face(mapping_point_idx_list)
        return mapping_face

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[Face]")
        print(line_start + "\t point_idx_list =", self.point_idx_list)
        return True

