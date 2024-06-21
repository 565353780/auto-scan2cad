#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import pi
import torch


def CIoU(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_z1, b1_x2, b1_y2, b1_z2 = box1
    b2_x1, b2_y1, b2_z1, b2_x2, b2_y2, b2_z2 = box2

    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0) * \
            (torch.min(b1_z2, b2_z2) - torch.max(b1_z1, b2_z1)).clamp(0)

    dx1, dy1, dz1 = b1_x2 - b1_x1, b1_y2 - b1_y1, b1_z2 - b1_z1
    dx2, dy2, dz2 = b2_x2 - b2_x1, b2_y2 - b2_y1, b2_z2 - b2_z1
    union = dx1 * dy1 * dz1 + dx2 * dy2 * dz2 - inter + eps

    iou = inter / union

    cdx = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    cdy = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    cdz = torch.max(b1_z2, b2_z2) - torch.min(b1_z1, b2_z1)
    c2 = cdx**2 + cdy**2 + cdz**2 + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2)**2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2)**2 +
            (b2_z1 + b2_z2 - b1_z1 - b1_z2)**2) / 9

    dx1, dy1, dz1 = dx1 + eps, dy1 + eps, dz1 + eps
    dx2, dy2, dz2 = dx2 + eps, dy2 + eps, dz2 + eps

    vxy = (4 / pi**2) * torch.pow(
        torch.atan(dx2 / dy2) - torch.atan(dx1 / dy1), 2)
    vyz = (4 / pi**2) * torch.pow(
        torch.atan(dy2 / dz2) - torch.atan(dy1 / dz1), 2)
    vzx = (4 / pi**2) * torch.pow(
        torch.atan(dz2 / dx2) - torch.atan(dz1 / dx1), 2)
    v = (vxy + vyz + vzx) / 3.0

    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)
