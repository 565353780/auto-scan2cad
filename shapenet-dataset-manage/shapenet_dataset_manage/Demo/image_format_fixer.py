#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shapenet_dataset_manage.Module.image_format_fixer import ImageFormatFixer

def demo():
    dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
    print_progress = True

    image_format_fixer = ImageFormatFixer(dataset_folder_path)
    image_format_fixer.fixAllImageFormat(print_progress)
    return True
