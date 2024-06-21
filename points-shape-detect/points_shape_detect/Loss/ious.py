#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from points_shape_detect.Loss.iou.ciou import CIoU
from points_shape_detect.Loss.iou.diou import DIoU
from points_shape_detect.Loss.iou.eiou import EIoU
from points_shape_detect.Loss.iou.giou import GIoU
from points_shape_detect.Loss.iou.iou import IoU


def checkShape(tensor1, tensor2):
    if tensor1.shape != tensor2.shape:
        return False
    if not 1 <= len(tensor1.shape) <= 2:
        return False
    return True


def getBatchIoU(iou_func, batch_box1, batch_box2, eps=1e-7):
    assert batch_box1.shape[0] == batch_box2.shape[0]

    batch_iou = torch.cat([
        (1.0 - iou_func(batch_box1[i], batch_box2[i])).reshape(1)
        for i in range(batch_box1.shape[0])
    ])
    return batch_iou


def getIoU(iou_func, box1, box2, eps=1e-7):
    assert checkShape(box1, box2)
    if len(box1.shape) == 1:
        return iou_func(box1, box2, eps)
    return getBatchIoU(iou_func, box1, box2, eps)


class IoULoss(object):

    @classmethod
    def IoU(cls, box1, box2, eps=1e-7):
        return getIoU(IoU, box1, box2, eps)

    @classmethod
    def CIoU(cls, box1, box2, eps=1e-7):
        return getIoU(CIoU, box1, box2, eps)

    @classmethod
    def DIoU(cls, box1, box2, eps=1e-7):
        return getIoU(DIoU, box1, box2, eps)

    @classmethod
    def EIoU(cls, box1, box2, eps=1e-7):
        return getIoU(EIoU, box1, box2, eps)

    @classmethod
    def GIoU(cls, box1, box2, eps=1e-7):
        return getIoU(GIoU, box1, box2, eps)
