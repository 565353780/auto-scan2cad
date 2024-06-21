#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

from points_shape_detect.Method.trans import transPointArray

from noc_transform.Data.obb import OBB

from noc_transform.Method.noc import getNOCOBB
from noc_transform.Method.transform import getNOCTransform


def testTransform(print_progress=False):
    test_num = 10000
    error_max = 1e-5

    for_data = range(test_num)
    if print_progress:
        print("[INFO][transform::testTransform]")
        print("\t start test transform error...")
        for_data = tqdm(for_data)
    for _ in for_data:
        obb = OBB.fromABBList([-0.4, -0.3, -0.5, 0.4, 0.3, 0.5])
        points = obb.points
        translate = np.random.rand(3)
        euler_angle = np.random.rand(3)
        scale = np.random.rand(3)

        points = transPointArray(points, translate, euler_angle, scale)
        obb.points = points

        noc_obb = getNOCOBB(obb)

        trans_matrix = getNOCTransform(obb)
        trans_matrix_inv = np.linalg.inv(trans_matrix)

        trans_obb = obb.clone()
        trans_obb.transform(trans_matrix)

        trans_error = trans_obb.points - noc_obb.points
        assert np.max(trans_error.reshape(-1)) < error_max

        noc_trans_obb = noc_obb.clone()
        noc_trans_obb.transform(trans_matrix_inv)

        trans_inv_error = obb.points - noc_trans_obb.points
        assert np.max(trans_inv_error.reshape(-1)) < error_max
    return True


def test():
    print_progress = True

    print("[INFO][transform::test]")
    print("\t start testTransform...")
    assert testTransform(print_progress)
    print("\t\t success!")

    return True
