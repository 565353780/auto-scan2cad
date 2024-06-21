#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


def GIoU(box1, box2, eps=1e-7):
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

    c_area = cdx * cdy * cdz + eps
    return iou - (c_area - union) / c_area
