#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Sample:
    filename: str = ""
    dimx: int = 0
    dimy: int = 0
    dimz: int = 0
    size: float = 0.0
    matrix: np.array = None
    tdf: np.array = None
    sign: np.array = None
