#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scan2cad_dataset_manage.Demo.dataset_loader import demo as demo_load_dataset
from scan2cad_dataset_manage.Demo.object_model_map_generator import demo as demo_generate_object_model_map
from scan2cad_dataset_manage.Demo.object_model_map_manager import demo as demo_manage_object_model_map

if __name__ == "__main__":
    demo_load_dataset()
    exit()
    demo_generate_object_model_map()
    demo_manage_object_model_map()
