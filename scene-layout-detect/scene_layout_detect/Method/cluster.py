#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scene_layout_detect.Data.line import Line
from scene_layout_detect.Method.dist import (fitLine, getPointDistToLine,
                                             getProjectPoint)
from scene_layout_detect.Method.render import renderPolyline

mode_list = ['cut', 'merge', 'angle']
mode = 'angle'


def getClusterEnergy(polylines, line_cluster_idx_list):
    cluster_energy = 0
    return cluster_energy


def getClusterIdxList(line_num, cut_idx_list):
    if cut_idx_list == []:
        return [0 for _ in range(line_num)]

    cluster_idx_list = []

    current_idx = 0
    for i in range(line_num):
        cluster_idx_list.append(current_idx)

        if i == max(cut_idx_list):
            current_idx = 0
            continue

        if i in cut_idx_list:
            current_idx += 1
    return cluster_idx_list


def getCutLineClusterIdxLists(polylines,
                              cut_num,
                              current_cut_idx_list=[],
                              start_idx=0):
    line_num = len(polylines)

    if cut_num == 0:
        cluster_idx_list = getClusterIdxList(line_num, current_cut_idx_list)
        return [cluster_idx_list]

    assert cut_num <= line_num - start_idx

    if cut_num == line_num - start_idx:
        cut_idx_list = current_cut_idx_list + [range(start_idx, line_num)]
        cluster_idx_list = getClusterIdxList(line_num, cut_idx_list)
        return [cluster_idx_list]

    cut_line_cluster_idx_lists = []

    for i in range(start_idx, line_num):
        remain_cut_position_num = line_num - i - 1
        if remain_cut_position_num < cut_num - 1:
            break

        new_cut_idx_list = current_cut_idx_list + [i]
        sub_cluster_idx_lists = getCutLineClusterIdxLists(
            polylines, cut_num - 1, new_cut_idx_list, start_idx + 1)
        cut_line_cluster_idx_lists += sub_cluster_idx_lists

    return cut_line_cluster_idx_lists


def clusterPolylinesByCut(polylines):
    for i in range(len(polylines)):
        line_cluster_idx_lists = getCutLineClusterIdxLists(polylines, i)
        print(line_cluster_idx_lists)
        print(np.array(line_cluster_idx_lists).shape)
        if i > 0:
            exit()

    exit()
    return line_cluster_idx_list


def getPointList(polylines, start_idx, end_idx):
    point_num = len(polylines)

    point_list = []
    for i in range(start_idx, end_idx):
        point_idx = i % point_num
        point_list.append(polylines[point_idx])
        if i == end_idx - 1:
            end_point_idx = end_idx % point_num
            point_list.append(polylines[end_point_idx])

    return point_list


def getLineParallelError(polylines, start_idx, end_idx):
    point_list = getPointList(polylines, start_idx, end_idx)
    line_param = fitLine(point_list)

    parallel_error = 0
    for point in point_list:
        parallel_error += getPointDistToLine(point, line_param)

    return parallel_error


def mergeLineByIdx(polylines, start_idx, end_idx):
    point_num = len(polylines)

    point_list = getPointList(polylines, start_idx, end_idx)
    line_param = fitLine(point_list)

    real_end_idx = (end_idx + 1) % point_num

    start_point = getProjectPoint(polylines[start_idx], line_param)
    end_point = getProjectPoint(polylines[real_end_idx], line_param)

    merged_polylines = []

    merged_polylines.append(start_point)
    merged_polylines.append(end_point)

    if real_end_idx < start_idx:
        for i in range(real_end_idx + 1, start_idx):
            merged_polylines.append(polylines[i])

        merged_polylines = np.array(merged_polylines, dtype=float)
        return merged_polylines

    for i in range(real_end_idx + 1, start_idx + point_num):
        current_real_idx = i % point_num
        merged_polylines.append(polylines[current_real_idx])

    merged_polylines = np.array(merged_polylines, dtype=float)
    return merged_polylines


def mergeLineByParallelError(polylines):
    point_num = len(polylines)

    min_error = float('inf')
    min_error_start_idx = None
    min_error_end_idx = None

    for i in range(point_num):
        #FIXME: tmp use j = i + 2, since this will always result on minimal error
        #  for j in range(2, point_num - 1):
        j = i + 2
        current_error = getLineParallelError(polylines, i, i + j)

        if current_error < min_error:
            min_error = current_error
            min_error_start_idx = i
            min_error_end_idx = i + j

    if min_error_start_idx is None:
        return polylines, min_error

    merged_polylines = mergeLineByIdx(polylines, min_error_start_idx,
                                      min_error_end_idx)
    return merged_polylines, min_error


def mergeAllLinesByParallelError(polylines):
    max_error = 4

    merged_polylines = np.array(polylines, dtype=float)

    while True:
        new_merged_polylines, min_error = mergeLineByParallelError(
            merged_polylines)

        if min_error > max_error:
            break

        if new_merged_polylines.shape[0] < 4:
            break

        merged_polylines = new_merged_polylines

    #  renderPolyline(merged_polylines, 'source')
    return merged_polylines


def clusterPolylinesByMerge(polylines):
    merged_polylines = mergeLineByParallelError(polylines)
    return merged_polylines


def clusterPolylinesByAngle(polylines):
    merged_polylines = mergeAllLinesByParallelError(polylines)
    return merged_polylines


def clusterPolylines(polylines):
    assert mode in mode_list

    if mode == 'cut':
        return clusterPolylinesByCut(polylines)
    if mode == 'merge':
        return clusterPolylinesByMerge(polylines)
    if mode == 'angle':
        return clusterPolylinesByAngle(polylines)
