#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy

import numpy as np
import open3d as o3d


def isPointInPolygon(polygon, point):
    '''
    Input:
    polygon: closed. i.e. polygon[0] == polygon[-1]
    Return:
        0 - the point is outside the polygon
        1 - the point is inside the polygon
        2 - the point is one edge (boundary)
    '''
    length = len(polygon) - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy * dy2 <= 0.0 and (point[0] >= polygon[ii][0]
                                or point[0] >= polygon[jj][0]):

            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy * (polygon[jj][0] -
                          polygon[ii][0]) / (dy - dy2) + polygon[ii][0]

                if point[0] > F:  # if line is left from the point - the ray moving towards left, will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 2

            # point on upper peak (dy2=dx2=0) or horizontal line (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (point[0] == polygon[jj][0] or
                               (dy == 0 and (point[0] - polygon[ii][0]) *
                                (point[0] - polygon[jj][0]) <= 0)):
                return 2

            # there is another posibility: (dy=0 and dy2>0) or (dy>0 and dy2=0). It is skipped
            # deliberately to prevent break-points intersections to be counted twice.

        ii = jj
        jj += 1

    return intersections & 1


def getDist2(point_1, point_2):
    x_diff = point_2[0] - point_1[0]
    y_diff = point_2[1] - point_1[1]
    return x_diff**2 + y_diff**2


def getMinDist2(point, point_list):
    min_dist = float('inf')
    for target_point in point_list:
        current_dist2 = getDist2(point, target_point)
        if current_dist2 < min_dist:
            min_dist = current_dist2
    return min_dist


def generateRegularMesh(floor_array):
    assert floor_array.shape[0] < 5

    if floor_array.shape[0] < 3:
        triangles = np.array([], dtype=int)

    if floor_array.shape[0] == 3:
        triangles = np.array([[0, 1, 2]], dtype=int)
    else:
        triangles = np.array([[0, 1, 2], [2, 3, 0]], dtype=int)

    regular_mesh = o3d.geometry.TriangleMesh()
    regular_mesh.vertices = o3d.utility.Vector3dVector(floor_array)
    regular_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    regular_mesh.compute_vertex_normals()
    return regular_mesh


def generateFloorMesh(floor_array, delta_dist=0.05):
    point_num = len(floor_array)

    if point_num < 5:
        return generateRegularMesh(floor_array)

    min_point_dist = delta_dist / 4

    closed_boundary = deepcopy(floor_array).tolist()
    closed_boundary.append(closed_boundary[0])

    min_point = np.min(floor_array, axis=0)
    max_point = np.max(floor_array, axis=0)

    diff = max_point - min_point

    diff_num = (diff / delta_dist).astype(int) + 2

    points = []
    for i in range(point_num):
        current_point = floor_array[i]
        next_idx = (i + 1) % point_num
        next_point = floor_array[next_idx]

        dist = np.linalg.norm(next_point - current_point)

        if dist == 0:
            if getMinDist2(current_point, points) > min_point_dist:
                points.append(current_point)
            continue

        sample_num = int(dist / delta_dist) + 2
        real_running_dist = dist / sample_num

        diff_point = (next_point - current_point) / dist * real_running_dist
        for i in range(sample_num):
            sample_point = current_point + i * diff_point
            if getMinDist2(sample_point, points) > min_point_dist:
                points.append(sample_point)

    for i in range(diff_num[0]):
        x = min_point[0] + i * delta_dist
        for j in range(diff_num[1]):
            y = min_point[1] + j * delta_dist

            current_point = [x, y, 0]

            if getMinDist2(current_point, points) < min_point_dist:
                continue

            state = isPointInPolygon(closed_boundary, current_point)
            if state != 1:
                continue

            points.append(current_point)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points, dtype=float))
    pcd.normals = o3d.utility.Vector3dVector(
        np.array([[0, 0, 1] for _ in range(len(points))], dtype=float))

    floor_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector(
            np.array([
                delta_dist, delta_dist * 2, delta_dist * 3, delta_dist * 4,
                delta_dist * 5
            ])))
    return floor_mesh


def generateLayoutMesh(floor_array,
                       wall_height=3,
                       top_floor_array=None,
                       expand_scale=1.0,
                       skip_floor=False):
    point_num = len(floor_array)

    if top_floor_array is None:
        top_floor_array = deepcopy(floor_array)
        top_floor_array[:, 2] += wall_height

    verts = np.vstack((floor_array, top_floor_array))

    triangles = []
    for i in range(point_num):
        next_idx = (i + 1) % point_num
        triangles.append([next_idx, i, next_idx + point_num])
        triangles.append([i + point_num, next_idx + point_num, i])

    if skip_floor:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.compute_vertex_normals()
        return mesh

    if expand_scale != 1.0:
        center = np.mean(floor_array, axis=0)
        floor_array -= center
        floor_array *= expand_scale
        floor_array += center

    floor_mesh = generateFloorMesh(floor_array)
    floor_verts = np.array(floor_mesh.vertices)
    floor_triangles = np.array(floor_mesh.triangles) + verts.shape[0]

    final_verts = np.vstack([verts, floor_verts])
    final_triangles = np.vstack([triangles, floor_triangles])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(final_verts)
    mesh.triangles = o3d.utility.Vector3iVector(final_triangles)
    mesh.compute_vertex_normals()
    return mesh
