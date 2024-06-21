#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import subprocess

def getImageFormat(image_file_path):
    assert os.path.exists(image_file_path)

    image_str = subprocess.check_output(['file', image_file_path]).decode()

    if "PNG image data" in image_str:
        return "png"
    if "JPEG image data" in image_str:
        return "jpg"
    return "empty"

def isImageFormatValid(image_file_path):
    image_format = getImageFormat(image_file_path)

    current_image_format = image_file_path.split(".")[-1]
    if current_image_format in ["jpg", "jpeg"]:
        return image_format in ["jpg", "jpeg"], image_format
    if current_image_format == "png":
        return image_format == "png", image_format
    return False, image_format
