#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime


def getCurrentTime():
    now = datetime.now()
    return now.strftime("%Y%m%d_%H:%M:%S")
