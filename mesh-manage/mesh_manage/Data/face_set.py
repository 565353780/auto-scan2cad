#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm

from mesh_manage.Data.face import Face

from mesh_manage.Method.pool import getFaceIdxListInPointIdxListWithPool

class FaceSet(object):
    def __init__(self):
        self.face_list = []
        return

    def reset(self):
        self.face_list = []
        return True

    def getFaceIdx(self, face):
        if len(self.face_list) == 0:
            return None

        for i in range(len(self.face_list)):
            if not self.face_list[i].isSameFace(face):
                continue
            return i

        return None

    def isHaveFace(self, face):
        face_idx = self.getFaceIdx(face)
        if face_idx is None:
            return False
        return True

    def addFace(self, point_idx_list, no_repeat=False):
        face = Face(point_idx_list)

        if no_repeat:
            if self.isHaveFace(face):
                return True

        self.face_list.append(face)
        return True

    def addFaceSet(self, face_set, point_start_idx):
        for face in face_set.face_list:
            point_idx_list = face.point_idx_list
            mapping_point_idx_list = [point_idx + point_start_idx for point_idx in point_idx_list]
            mapping_face = Face(mapping_point_idx_list)
            self.face_list.append(mapping_face)
        return True

    def getFace(self, face_idx):
        if face_idx >= len(self.face_list):
            print("[ERROR][FaceSet::getFace]")
            print("\t face_idx out of range!")
            return None
        return self.face_list[face_idx]

    def getPointIdxListAndMappingDict(self, face_idx_list):
        point_idx_list = []
        mapping_dict = {}

        for face_idx in face_idx_list:
            face = self.getFace(face_idx)
            if face is None:
                print("[ERROR][FaceSet::getPointIdxListAndMappingDict]")
                print("\t getFace failed!")
                return None, None

            for point_idx in face.point_idx_list:
                if point_idx in point_idx_list:
                    continue
                mapping_dict[str(point_idx)] = len(point_idx_list)
                point_idx_list.append(point_idx)

        return point_idx_list, mapping_dict

    def getMappingFaceSet(self, face_idx_list, mapping_dict):
        mapping_face_set = FaceSet()
        for face_idx in face_idx_list:
            face = self.getFace(face_idx)
            if face is None:
                print("[ERROR][FaceSet::getMappingFaceSet]")
                print("\t getFace failed!")
                return None

            mapping_point_idx_list = face.getMappingPointIdxList(mapping_dict)
            if mapping_point_idx_list is None:
                print("[ERROR][FaceSet::getMappingFaceSet]")
                print("\t getMappingPointIdxList failed!")
                return None

            mapping_face_set.addFace(mapping_point_idx_list)
        return mapping_face_set

    def getFaceIdxListInPointIdxList(self, point_idx_list):
        face_idx_list = []
        print("[INFO][FaceSet::getFaceIdxListInPointIdxList]")
        print("\t start check if is face in point_idx_list...")
        for i, face in tqdm(enumerate(self.face_list), total=len(self.face_list)):
            if face.isInPointIdxList(point_idx_list):
                face_idx_list.append(i)
        return face_idx_list

    def getFaceIdxListInPointIdxListWithPool(self, point_idx_list):
        return getFaceIdxListInPointIdxListWithPool(self.face_list, point_idx_list)

    def getFaceSetInPointIdxList(self, point_idx_list):
        #  face_idx_list = self.getFaceIdxListInPointIdxList(point_idx_list)
        face_idx_list = self.getFaceIdxListInPointIdxListWithPool(point_idx_list)

        mapping_dict = {}
        used_point_idx_list = []
        for point_idx in point_idx_list:
            if point_idx in used_point_idx_list:
                continue
            mapping_dict[str(point_idx)] = len(used_point_idx_list)
            used_point_idx_list.append(point_idx)

        face_set = self.getMappingFaceSet(face_idx_list, mapping_dict)
        if face_set is None:
            print("[ERROR][FaceSet::getFaceSetInPointIdxList]")
            print("\t getMappingFaceSet failed!")
            return None
        return face_set

    def getPointIdxListList(self):
        point_idx_list_list = [face.point_idx_list for face in self.face_list]
        return point_idx_list_list

    def outputInfo(self, info_level=0):
        line_start = "\t" * info_level
        print(line_start + "[FaceSet]")
        print(line_start + "\t face_num =", len(self.face_list))
        return True

