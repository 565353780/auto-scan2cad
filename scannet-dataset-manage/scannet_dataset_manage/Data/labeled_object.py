#!/usr/bin/env python
# -*- coding: utf-8 -*-


class LabeledObject(object):

    def __init__(self,
                 id=None,
                 object_id=None,
                 segment_idx_list=None,
                 label=None,
                 object_dict=None):
        self.id = id
        self.object_id = object_id
        self.segment_idx_list = segment_idx_list
        self.label = label

        if object_dict is not None:
            self.loadDict(object_dict)
        return

    def loadDict(self, object_dict):
        self.id = object_dict["id"]
        self.object_id = object_dict["objectId"]
        self.segment_idx_list = object_dict["segments"]
        self.label = object_dict["label"]
        return True
