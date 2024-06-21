#!/usr/bin/env python
# -*- coding: utf-8 -*-

from shapenet_dataset_manage.Data.synset import Synset


class SynsetLoader(object):

    def __init__(self):
        self.synset = Synset()
        return

    def reset(self):
        self.synset.reset()
        return True

    def loadSynset(self, synset_root_path):
        assert self.synset.loadRootPath(synset_root_path)
        return True

    def outputInfo(self, info_level=0, print_cols=5):
        self.synset.outputInfo(info_level, print_cols)
        return True
