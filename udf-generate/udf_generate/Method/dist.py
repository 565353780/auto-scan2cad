#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def getUDFDist(udf_1, udf_2, ord=2):
    udf_1_array = np.reshape(udf_1, (-1))
    udf_2_array = np.reshape(udf_2, (-1))
    return np.linalg.norm(udf_1_array - udf_2_array, ord=ord)
