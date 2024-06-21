#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

LINES = np.array([[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6],
                  [6, 4], [0, 4], [1, 5], [3, 7], [2, 6]])

X_AXIS_IDX = np.array([8, 9, 10, 11])
Y_AXIS_IDX = np.array([1, 3, 5, 7])
Z_AXIS_IDX = np.array([0, 2, 4, 6])

X_AXIS_LINES = LINES[X_AXIS_IDX]
Y_AXIS_LINES = LINES[Y_AXIS_IDX]
Z_AXIS_LINES = LINES[Z_AXIS_IDX]
