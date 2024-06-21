#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


# batch*n
def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(
        v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    #print (u.shape)
    #print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)),
                    1)  #batch*3

    return out


#poses batch*6
#poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  #batch*3
    y_raw = poses[:, 3:6]  #batch*3

    x = normalize_vector(x_raw)  #batch*3
    z = cross_product(x, y_raw)  #batch*3
    z = normalize_vector(z)  #batch*3
    y = cross_product(z, x)  #batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  #batch*3*3
    return matrix


#matrices batch*3*3
#both matrix are orthogonal rotation matrices
#out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch = m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1, 2))  #batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos,
                    torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    #theta = torch.min(theta, 2*np.pi - theta)

    return theta


#axisAngle batch*3*3s angle, x,y,z
def get_sampled_rotation_matrices_by_axisAngle(batch):

    theta = torch.autograd.Variable(
        torch.FloatTensor(np.random.uniform(-1, 1, batch) *
                          np.pi).cuda())  #[0, pi] #[-180, 180]
    sin = torch.sin(theta)

    #  axis = torch.autograd.Variable(torch.randn(batch, 3).cuda())
    # FIXME: set only rotate via z-axis
    axis = torch.autograd.Variable(torch.zeros(batch, 3).cuda())
    axis[:, 1] = 1.0

    axis = normalize_vector(axis)  #batch*3
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 * zw, 2 * xz + 2 * yw),
                     1)  #batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 * zz, 2 * yz - 2 * xw),
                     1)  #batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw, 1 - 2 * xx - 2 * yy),
                     1)  #batch*3

    matrix = torch.cat(
        (row0.view(batch, 1, 3), row1.view(batch, 1, 3), row2.view(
            batch, 1, 3)), 1)  #batch*3*3

    return matrix
