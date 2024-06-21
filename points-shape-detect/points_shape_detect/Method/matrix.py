#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import quaternion
import numpy as np


def make_M_from_tqs(t: list, q: list, s: list, center=None) -> np.ndarray:
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M


def getXRotateMatrixTensor(rotate_angle):
    device = rotate_angle.device

    rotate_rad = rotate_angle * np.pi / 180.0

    x_rotate_matrix_tensor = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0,
                           torch.cos(rotate_rad), -torch.sin(rotate_rad)],
         [0.0, torch.sin(rotate_rad),
          torch.cos(rotate_rad)]]).to(device)
    return x_rotate_matrix_tensor


def getXRotateMatrix(rotate_angle):
    rotate_rad = rotate_angle * np.pi / 180.0

    x_rotate_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0,
                                 np.cos(rotate_rad), -np.sin(rotate_rad)],
                                [0.0,
                                 np.sin(rotate_rad),
                                 np.cos(rotate_rad)]])
    return x_rotate_matrix


def getYRotateMatrixTensor(rotate_angle):
    device = rotate_angle.device

    rotate_rad = rotate_angle * np.pi / 180.0

    y_rotate_matrix_tensor = torch.tensor(
        [[torch.cos(rotate_rad), 0.0,
          torch.sin(rotate_rad)], [0.0, 1.0, 0.0],
         [-torch.sin(rotate_rad), 0.0,
          torch.cos(rotate_rad)]]).to(device)
    return y_rotate_matrix_tensor


def getYRotateMatrix(rotate_angle):
    rotate_rad = rotate_angle * np.pi / 180.0

    y_rotate_matrix = np.array([[np.cos(rotate_rad), 0.0,
                                 np.sin(rotate_rad)], [0.0, 1.0, 0.0],
                                [-np.sin(rotate_rad), 0.0,
                                 np.cos(rotate_rad)]])
    return y_rotate_matrix


def getZRotateMatrixTensor(rotate_angle):
    device = rotate_angle.device

    rotate_rad = rotate_angle * np.pi / 180.0

    z_rotate_matrix_tensor = torch.tensor(
        [[torch.cos(rotate_rad), -torch.sin(rotate_rad), 0.0],
         [torch.sin(rotate_rad),
          torch.cos(rotate_rad), 0.0], [0.0, 0.0, 1.0]]).to(device)
    return z_rotate_matrix_tensor


def getZRotateMatrix(rotate_angle):
    rotate_rad = rotate_angle * np.pi / 180.0

    z_rotate_matrix = np.array([[np.cos(rotate_rad), -np.sin(rotate_rad), 0.0],
                                [np.sin(rotate_rad),
                                 np.cos(rotate_rad), 0.0], [0.0, 0.0, 1.0]])
    return z_rotate_matrix


def getRotateMatrixTensor(xyz_rotate_angle, is_inverse=False):
    x_rotate_matrix_tensor = getXRotateMatrixTensor(xyz_rotate_angle[0])
    y_rotate_matrix_tensor = getYRotateMatrixTensor(xyz_rotate_angle[1])
    z_rotate_matrix_tensor = getZRotateMatrixTensor(xyz_rotate_angle[2])

    if is_inverse:
        return torch.matmul(
            torch.matmul(z_rotate_matrix_tensor, y_rotate_matrix_tensor),
            x_rotate_matrix_tensor)
    return torch.matmul(
        torch.matmul(x_rotate_matrix_tensor, y_rotate_matrix_tensor),
        z_rotate_matrix_tensor)


def getRotateMatrix(xyz_rotate_angle, is_inverse=False):
    if isinstance(xyz_rotate_angle, torch.Tensor):
        return getRotateMatrixTensor(xyz_rotate_angle, is_inverse)

    x_rotate_matrix = getXRotateMatrix(xyz_rotate_angle[0])
    y_rotate_matrix = getYRotateMatrix(xyz_rotate_angle[1])
    z_rotate_matrix = getZRotateMatrix(xyz_rotate_angle[2])

    if is_inverse:
        return z_rotate_matrix @ y_rotate_matrix @ x_rotate_matrix
    return x_rotate_matrix @ y_rotate_matrix @ z_rotate_matrix
