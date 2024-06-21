#!/usr/bin/env python
# -*- coding: utf-8 -*-

def outputList(data_list, info_level=0, print_cols=10):
    if print_cols < 1:
        print_cols = 1

    line_start = "\t" * info_level

    print(line_start + "[")
    print_num = 0
    for data in data_list:
        if print_num == 0:
            print(line_start + "\t", end="")
        print(data + ", ", end="")
        print_num += 1
        if print_num == print_cols:
            print()
            print_num = 0
    if print_num != 0:
        print()
    print(line_start + "]")
    return True

def outputJson(data_json, info_level=0):
    line_start = "\t" * info_level

    if data_json is None:
        print(line_start + "None")
        return True

    print(line_start + "{")
    for key in data_json.keys():
        print(line_start + "\t " + key + ": ", data_json[key])
    print(line_start + "}")
    return True

