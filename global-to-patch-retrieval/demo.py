#!/usr/bin/env python
# -*- coding: utf-8 -*-

from global_to_patch_retrieval.Demo.trainer import demo as demo_train
from global_to_patch_retrieval.Demo.retrieval_manager import demo as demo_manage_retrieval
from global_to_patch_retrieval.Demo.s2c_retrieval_manager import demo as demo_manage_s2c_retrieval
from global_to_patch_retrieval.Demo.detector import demo as demo_detect

if __name__ == "__main__":
    #  demo_train()
    #  demo_manage_retrieval()
    demo_manage_s2c_retrieval()
    #  demo_detect()
