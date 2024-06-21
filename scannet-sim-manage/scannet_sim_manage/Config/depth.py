#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

HFOV = 90. * np.pi / 180.
W = 480
H = 360

K = np.array([
    [1 / np.tan(HFOV / 2.), 0., 0., 0.],
    [0., 1 / np.tan(HFOV / 2.) * W / H, 0., 0.],
    [0., 0., 1, 0],
    [0., 0., 0, 1],
])

K_INV = np.linalg.inv(K)

XS, YS = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
XS = XS.reshape(1, W, H)
YS = YS.reshape(1, W, H)

INF = float('inf')
INF_POINT = [INF, INF, INF]
