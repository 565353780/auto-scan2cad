#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shapenet_dataset_manage.Module.Core.V2.synset_loader import SynsetLoader


def demo():
    synset_root_path = "/home/chli/scan2cad/shapenet/ShapeNetCore.v2/02691156/"

    synset_loader = SynsetLoader()
    synset_loader.loadSynset(synset_root_path)
    synset_loader.outputInfo()
    return True
