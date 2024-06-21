#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

from points_shape_detect.Loss.ious import IoULoss


def testArray(print_progress=False):
    test_num = 10

    bbox1 = np.array([0, 0, 0, 1, 1, 1])
    bbox2 = np.array([0.2, 0.2, 0.2, 1.2, 1.2, 1.2])
    bbox1 = torch.from_numpy(bbox1).float()
    bbox2 = torch.from_numpy(bbox2).float()

    for i in range(test_num):
        bbox2 = bbox1.clone()
        bbox2[5] = i

        if print_progress:
            print("====testArray : " + str(i + 1) + "/" + str(test_num) +
                  "====")
            iou = IoULoss.IoU(bbox1, bbox2)
            print("IoU", iou)
            iou = IoULoss.CIoU(bbox1, bbox2)
            print("CIoU", iou)
            iou = IoULoss.DIoU(bbox1, bbox2)
            print("DIoU", iou)
            iou = IoULoss.EIoU(bbox1, bbox2)
            print("EIoU", iou)
            iou = IoULoss.GIoU(bbox1, bbox2)
            print("GIoU", iou)
    return True


def testBatch(print_progress=False):
    test_num = 10

    batch_bbox1 = np.array([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    batch_bbox2 = np.array([[0.2, 0.2, 0.2, 1.2, 1.2, 1.2],
                            [0.2, 0.2, 0.2, 1.2, 1.2, 1.2]])
    batch_bbox1 = torch.from_numpy(batch_bbox1).float()
    batch_bbox2 = torch.from_numpy(batch_bbox2).float()

    for i in range(test_num):
        batch_bbox2 = batch_bbox1.clone()
        batch_bbox2[0][5] = i
        batch_bbox2[1][5] = i

        if print_progress:
            print("====testBatch : " + str(i + 1) + "/" + str(test_num) +
                  "====")
            iou = IoULoss.IoU(batch_bbox1, batch_bbox2)
            print("IoU", iou)
            iou = IoULoss.CIoU(batch_bbox1, batch_bbox2)
            print("CIoU", iou)
            iou = IoULoss.DIoU(batch_bbox1, batch_bbox2)
            print("DIoU", iou)
            iou = IoULoss.EIoU(batch_bbox1, batch_bbox2)
            print("EIoU", iou)
            iou = IoULoss.GIoU(batch_bbox1, batch_bbox2)
            print("GIoU", iou)
    return True


def test():
    print_progress = False

    print("[INFO][iou::test] start testArray...")
    assert testArray(print_progress)
    print("\t passed!")

    print("[INFO][iou::test] start testBatch...")
    assert testBatch(print_progress)
    print("\t passed!")
    return True
