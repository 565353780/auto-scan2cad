#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def isListInList(target_list, total_list):
    return all(np.isin(target_list, total_list))

