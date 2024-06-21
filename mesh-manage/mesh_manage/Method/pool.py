#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from multiprocessing import Pool

from mesh_manage.Data.channel_point import ChannelPoint


def isFaceInPointIdxList(inputs):
    face, point_idx_list = inputs
    return face.isInPointIdxList(point_idx_list)


def getFaceIdxListInPointIdxListWithPool(face_list, point_idx_list):
    inputs_list = [[face, point_idx_list] for face in face_list]

    pool = Pool(processes=os.cpu_count())
    result = pool.map(isFaceInPointIdxList, inputs_list)
    pool.close()
    pool.join()

    face_idx_list = np.where(np.array(result) == True)[0].tolist()
    return face_idx_list


def getChannelPoint(inputs):
    channel_name_list, channel_value_list = inputs
    return ChannelPoint(channel_name_list, channel_value_list)


def getChannelPointListWithPool(channel_name_list, channel_value_list_list):
    inputs_list = [[channel_name_list, channel_value_list]
                   for channel_value_list in channel_value_list_list]

    pool = Pool(processes=os.cpu_count())
    result = pool.imap(getChannelPoint, inputs_list, chunksize=64)
    pool.close()
    pool.join()

    channel_point_list = list(result)
    return channel_point_list
