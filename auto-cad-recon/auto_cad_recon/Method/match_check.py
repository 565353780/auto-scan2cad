#!/usr/bin/env python
# -*- coding: utf-8 -*-


def isMatch(source_name, target_name):
    source_label = source_name.split(".")[0].split("_")[1]
    target_label = target_name.split(".")[0].split("_")[1]
    return source_label in target_label or target_label in source_label
