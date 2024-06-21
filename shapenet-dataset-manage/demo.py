#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shapenet_dataset_manage.Demo.Core.V2.model_loader import demo as demo_load_model
from shapenet_dataset_manage.Demo.Core.V2.synset_loader import demo as demo_load_synset
from shapenet_dataset_manage.Demo.Core.V2.dataset_loader import demo as demo_load_dataset
from shapenet_dataset_manage.Demo.image_format_fixer import demo as demo_fix_image_format

if __name__ == "__main__":
    #  demo_load_model()
    #  demo_load_synset()
    #  demo_load_dataset()
    demo_fix_image_format()
