#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../image-to-cad")

import torch
import torch.nn as nn

from auto_cad_recon.Model.alignment.alignment_head import AlignmentHead
from auto_cad_recon.Model.retrieval.retrieval_head import RetrievalHead


class RetrievalNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda")

        self.alignment_head = AlignmentHead()
        self.retrieval_head = RetrievalHead()
        return

    def forward(self, data, print_progress=False):
        data = self.alignment_head(data)
        data = self.retrieval_head(data, print_progress)
        return data
