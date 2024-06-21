#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm

from points_shape_detect.Method.trans import \
    normalizePointArray, randomTransPointArray, \
    getInverseTrans, transPointArray
from points_shape_detect.Method.render import renderPointArrayWithUnitBBox

from points_shape_detect.Method.matrix import getRotateMatrix


def testMatrix(print_progress=False):
    test_num = 10000

    error = 1e-10

    for_data = range(test_num)
    if print_progress:
        for_data = tqdm(for_data)
    for _ in for_data:
        euler_angle = (np.random.rand(3) - 0.5) * 360.0
        euler_angle_inv = -1.0 * euler_angle
        rotate_matrix = getRotateMatrix(euler_angle)
        rotate_matrix_inv = getRotateMatrix(euler_angle_inv, True)
        rotate_matrix_inv2 = np.linalg.inv(rotate_matrix_inv)
        assert np.linalg.norm(rotate_matrix - rotate_matrix_inv2) < error
    return True


def testScale(print_progress=False):
    test_num = 100

    error = 1e-10

    for_data = range(test_num)
    if print_progress:
        for_data = tqdm(for_data)
    for _ in for_data:
        scale = np.random.rand(3) + 0.5
        scale_inv = 1.0 / scale

        point_array = np.random.randn(8192, 3)

        scale_point_array = point_array * scale

        scale_back_point_array = scale_point_array * scale_inv

        assert np.linalg.norm(point_array - scale_back_point_array) < error
    return True


def testEuler(print_progress=False):
    test_num = 100

    error = 1e-10

    for_data = range(test_num)
    if print_progress:
        for_data = tqdm(for_data)
    for _ in for_data:
        point_array = np.random.randn(8192, 3)

        normalize_point_array = normalizePointArray(point_array)

        random_point_array, translate, euler_angle, scale = randomTransPointArray(
            normalize_point_array, True)

        translate_inv, euler_angle_inv, scale_inv = getInverseTrans(
            translate, euler_angle, scale)
        move_back_point_array = transPointArray(random_point_array,
                                                translate_inv, euler_angle_inv,
                                                scale_inv, True)

        translate_inv2, euler_angle_inv2, scale_inv2 = getInverseTrans(
            translate_inv, euler_angle_inv, scale_inv)

        assert np.linalg.norm(translate_inv2 - translate) < error
        assert np.linalg.norm(euler_angle_inv2 - euler_angle) < error
        assert np.linalg.norm(scale_inv2 - scale) < error

        assert np.linalg.norm(move_back_point_array - normalize_point_array
                              ) < error * point_array.shape[0]
    return True


def testEulerTensor(print_progress=False):
    test_num = 100

    error = 1e-6

    for_data = range(test_num)
    if print_progress:
        for_data = tqdm(for_data)
    for _ in for_data:

        npy_file_path = "/home/chli/chLi/PoinTr/ShapeNet55/shapenet_pc/" + \
            "04090263-2eb7e88f5355630962a5697e98a94be.npy"
        point_array = np.load(npy_file_path)

        #  point_array = np.random.randn(8192, 3)

        point_array_tensor = torch.from_numpy(point_array.astype(
            np.float32)).cuda()
        normalize_point_array_tensor = normalizePointArray(point_array_tensor)

        random_point_array_tensor, translate, euler_angle, scale = randomTransPointArray(
            normalize_point_array_tensor, True)

        translate_inv, euler_angle_inv, scale_inv = getInverseTrans(
            translate, euler_angle, scale)
        move_back_point_array_tensor = transPointArray(
            random_point_array_tensor, translate_inv, euler_angle_inv,
            scale_inv, True)

        translate_inv2, euler_angle_inv2, scale_inv2 = getInverseTrans(
            translate_inv, euler_angle_inv, scale_inv)

        translate_error = torch.norm(translate_inv2 - translate).item()
        euler_angle_error = torch.norm(euler_angle_inv2 - euler_angle).item()
        scale_error = torch.norm(scale_inv2 - scale).item()

        assert translate_error < error, str(translate_error) + " > " + str(
            error)
        assert euler_angle_error < error, str(euler_angle_error) + " > " + str(
            error)
        assert scale_error < error, str(scale_error) + " > " + str(error)

        #  renderPointArrayWithUnitBBox(
        #  torch.vstack(
        #  (normalize_point_array_tensor, move_back_point_array_tensor +
        #  torch.tensor([0, 0, 1]).to(torch.float32).cuda())))

        points_error = torch.norm(move_back_point_array_tensor -
                                  normalize_point_array_tensor).item()
        assert points_error < error * point_array.shape[0], str(
            points_error) + " > " + str(error)
    return True


def test():
    print_progress = True

    print("[INFO][trans::test] start testMatrix...")
    assert testMatrix(print_progress)
    print("\t passed!")

    print("[INFO][trans::test] start testScale...")
    assert testScale(print_progress)
    print("\t passed!")

    print("[INFO][trans::test] start testEuler...")
    assert testEuler(print_progress)
    print("\t passed!")

    print("[INFO][trans::test] start testEulerTensor...")
    assert testEulerTensor(print_progress)
    print("\t passed!")
    return True
