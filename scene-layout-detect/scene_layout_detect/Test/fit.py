#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scene_layout_detect.Method.dist import fitLine, getPointDistToLine


def outputLine(line_param):
    A, B, C = line_param
    if A is None:
        print('this line is a single point')
    else:
        print(str(A) + ' x + ' + str(B) + ' y + ' + str(C) + ' = 0')
    return True


def testPointList(point_list):
    test_point = [0, 0]

    line_param = fitLine(point_list)
    outputLine(line_param)

    dist = getPointDistToLine(test_point, line_param)
    print("dist =", dist)
    return True


def test():
    testPointList([[0, 1], [1, 2], [2, 3]])
    testPointList([[322, 74], [304, 63], [303, 56]])
    testPointList([[0, 0], [1, 0], [2, 0]])
    testPointList([[0, 0], [0, 1], [0, 2]])
    testPointList([[2, 2], [2, 2], [2, 2]])
    return True
