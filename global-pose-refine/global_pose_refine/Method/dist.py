#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
from scipy.spatial import ConvexHull


def getPointDist2(point_1, point_2):
    x_diff = point_1[0] - point_2[0]
    y_diff = point_1[1] - point_2[1]
    z_diff = point_1[2] - point_2[2]

    dist2 = x_diff**2 + y_diff**2 + z_diff**2
    return dist2


def getPointDist(point_1, point_2):
    dist2 = getPointDist2(point_1, point_2)
    dist = np.sqrt(dist2)
    return dist


def getOBBCenter(obb):
    return (obb.points[0] + obb.points[7]) / 2.0


def getOBBEdgeLengthList(obb):
    x_length = getPointDist(obb.points[0], obb.points[4])
    y_length = getPointDist(obb.points[0], obb.points[2])
    z_length = getPointDist(obb.points[0], obb.points[1])
    return [x_length, y_length, z_length]


def getOBBMinLengthToOut(obb):
    length_list = getOBBEdgeLengthList(obb)
    min_length = np.min(length_list) / 2.0
    return min_length


def getOBBMaxLengthToOut(obb):
    center = getOBBCenter(obb)
    dist_list = [getPointDist(point, center) for point in obb.points]
    max_length = np.max(dist_list)
    return max_length


def isOBBsAway(obb_1, obb_2):
    obb1_points = obb_1.points[:, :2]
    obb2_points = obb_2.points[:, :2]

    min_point_1 = np.min(obb1_points, axis=0)
    max_point_1 = np.max(obb1_points, axis=0)
    min_point_2 = np.min(obb2_points, axis=0)
    max_point_2 = np.max(obb2_points, axis=0)

    if max_point_1[0] < min_point_2[0]:
        return True
    if max_point_1[1] < min_point_2[1]:
        return True

    if max_point_2[0] < min_point_1[0]:
        return True
    if max_point_2[1] < min_point_1[1]:
        return True
    return False


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


def isOBBPointInOBB(obb_1, obb_2):
    obb2_points = obb_2.points[:, :2]
    hull = ConvexHull(obb2_points)
    polygon = obb2_points[hull.vertices]
    polygon = np.vstack([polygon, [polygon[0]]])

    for point in obb_1.points:
        if isPointInPolygon(polygon, point) == 1:
            return True
    return False


def cross(p1, p2, p3):
    x1 = p2[0] - p1[0]
    y1 = p2[1] - p1[1]
    x2 = p3[0] - p1[0]
    y2 = p3[1] - p1[1]
    return x1 * y2 - x2 * y1


def isIntersec(p1, p2, p3, p4):
    if (max(p1[0], p2[0]) >= min(p3[0], p4[0])
            and max(p3[0], p4[0]) >= min(p1[0], p2[0])
            and max(p1[1], p2[1]) >= min(p3[1], p4[1])
            and max(p3[1], p4[1]) >= min(p1[1], p2[1])):

        if (cross(p1, p2, p3) * cross(p1, p2, p4) <= 0
                and cross(p3, p4, p1) * cross(p3, p4, p2) <= 0):
            D = 1
        else:
            D = 0
    else:
        D = 0
    return D


def isLineCrossOBB(start, end, obb):
    obb_point_num = obb.points.shape[0]
    for i in range(obb_point_num):
        next_idx = (i + 1) % obb_point_num

        if isIntersec(start, end, obb.points[i], obb.points[next_idx]):
            return True
    return False


def isOBBCrossOBB(obb_1, obb_2):
    obb1_point_num = obb_1.points.shape[0]
    for i in range(obb1_point_num):
        next_idx = (i + 1) % obb1_point_num

        if isLineCrossOBB(obb_1.points[i], obb_1.points[next_idx], obb_2):
            return True
    return False


def isOBBCrossOnZ(obb_1, obb_2):
    if isOBBsAway(obb_1, obb_2):
        return False

    if isOBBPointInOBB(obb_1, obb_2):
        return True

    if isOBBPointInOBB(obb_2, obb_1):
        return True

    center_1 = getOBBCenter(obb_1)
    center_2 = getOBBCenter(obb_2)
    center_dist = getPointDist(center_1, center_2)

    min_dist = getOBBMinLengthToOut(obb_1) + getOBBMinLengthToOut(obb_2)
    if center_dist <= min_dist:
        return True

    max_dist = getOBBMaxLengthToOut(obb_1) + getOBBMaxLengthToOut(obb_2)
    if center_dist > max_dist:
        return False

    if isOBBCrossOBB(obb_1, obb_2):
        return True

    if isOBBCrossOBB(obb_2, obb_1):
        return True

    return False


def getOBBDistZ(obb_low, obb_high):
    lower_z = np.max(obb_low.points[:, 2])
    higher_z = np.min(obb_high.points[:, 2])

    dist_z = np.abs(lower_z - higher_z)
    return dist_z


def getOBBSupportDist(obb_1, obb_2):
    if not isOBBCrossOnZ(obb_1, obb_2):
        return 0

    obb_center_1 = getOBBCenter(obb_1)
    obb_center_2 = getOBBCenter(obb_2)
    if obb_center_1[2] < obb_center_2[2]:
        return getOBBDistZ(obb_1, obb_2)
    return getOBBDistZ(obb_2, obb_1)


def getOBBDirection(obb):
    start = obb.points[0]
    x_end = obb.points[4]
    y_end = obb.points[2]
    z_end = obb.points[1]

    x_direction = x_end - start
    y_direction = y_end - start
    z_direction = z_end - start

    x_norm = np.linalg.norm(x_direction)
    y_norm = np.linalg.norm(y_direction)
    z_norm = np.linalg.norm(z_direction)

    if x_norm > 0:
        x_direction /= x_norm
    if y_norm > 0:
        y_direction /= y_norm
    if z_norm > 0:
        z_direction /= z_norm
    return [x_direction, y_direction, z_direction]


def getDirectionDist(obb_1, obb_2):
    direction1_list = getOBBDirection(obb_1)
    dx2, dy2, dz2 = getOBBDirection(obb_2)

    min_direction_dist = float('inf')

    for dx, dy, dz in itertools.product([dx2, -dx2], [dy2, -dy2], [dz2, -dz2]):
        for x_idx, y_idx, z_idx in itertools.permutations([0, 1, 2], 3):
            dx1_new = direction1_list[x_idx]
            current_x_dist = np.linalg.norm(dx1_new - dx)
            dy1_new = direction1_list[y_idx]
            current_y_dist = np.linalg.norm(dy1_new - dy)
            dz1_new = direction1_list[z_idx]
            current_z_dist = np.linalg.norm(dz1_new - dz)

            current_dist = current_x_dist + current_y_dist + current_z_dist
            if current_dist < min_direction_dist:
                min_direction_dist = current_dist

    return min_direction_dist


def getOBBPoseDist(obb_1, obb_2):
    obb_center_1 = getOBBCenter(obb_1)
    obb_center_2 = getOBBCenter(obb_2)
    center_dist = getPointDist(obb_center_1, obb_center_2)

    direction_dist = getDirectionDist(obb_1, obb_2)

    pose_dist = 1.0 / (direction_dist + center_dist + 1e-6)
    return pose_dist
