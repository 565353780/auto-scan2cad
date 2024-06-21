#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import random
import numpy as np
from collections import defaultdict

from torch.utils.data import Dataset

from global_to_patch_retrieval.Method.sample import load_raw_df, load_mask, load_sdf
from global_to_patch_retrieval.Method.transform import \
    to_occupancy_grid, truncation_normalization_transform
from global_to_patch_retrieval.Method.augment import \
    rotation_augmentation_interpolation_v2, rotation_augmentation_fixed, \
    flip_augmentation, jitter_augmentation


class Scan2CAD(Dataset):

    def __init__(self,
                 dataset_file,
                 scannet_root,
                 shapenet_root,
                 splits=["train"]):
        super().__init__()

        # in [train, validation, test]
        self.splits = splits
        self.scannet_root = scannet_root
        self.shapenet_root = shapenet_root
        self.dataset_file = dataset_file

        # in [fixed, interpolation, none]
        self.rotation_augmentation = "interpolation"
        self.mask_scans = False

        # in [truncation_normalization_transform, to_occupancy_grid]
        self.transformation = to_occupancy_grid

        self.pairs = self.load_from_json(self.dataset_file)

        self.negatives = self.add_negatives(self.pairs)
        return

    def load_from_json(self, file):
        with open(file) as f:
            content = json.load(f)
            objects = content["scan2cad_objects"]

            pairs = []

            for k, v in objects.items():
                if v in self.splits:
                    pair = (k, k)
                    pairs.append(pair)

        return pairs

    @staticmethod
    def add_negatives(data):
        per_category = defaultdict(list)

        for scan, _ in data:
            category = scan.split("_")[4]
            per_category[category].append(scan)

        negatives = []
        for scan, _ in data:
            category = scan.split("_")[4]
            neg_categories = list(per_category.keys())
            neg_categories.remove(category)
            neg_category = np.random.choice(neg_categories)
            neg_cad = np.random.choice(per_category[neg_category])
            negatives.append(neg_cad)

        return negatives

    def regenerate_negatives(self):
        self.negatives = self.add_negatives(self.pairs)
        return True

    def __getitem__(self, index):
        objects = {}

        # Load scan sample
        scannet_object_name, shapenet_object_name = self.pairs[index]
        scannet_object_path = self.scannet_root + scannet_object_name + ".sdf"

        # Load scan mask
        scannet_mask_path = self.scannet_root + scannet_object_name + ".mask"
        objects["mask"], _ = self._load(scannet_mask_path)

        objects["scan"], _ = self._load(scannet_object_path, scannet_mask_path)

        # Load CAD sample
        shapenet_object_path = self.shapenet_root + shapenet_object_name + ".df"
        objects["cad"], _ = self._load(shapenet_object_path)

        # Load negative CAD sample
        negative_name = self.negatives[index]
        negative_object_path = self.shapenet_root + negative_name + ".df"
        objects["negative"], _ = self._load(negative_object_path)

        # Apply augmentations
        if self.rotation_augmentation == "interpolation":
            rotations = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
            degree = random.choice(rotations)
            angle = degree * math.pi / 180
            objects = {
                k: rotation_augmentation_interpolation_v2(o, angle)
                for k, o in objects.items()
            }

        elif self.rotation_augmentation == "fixed":
            objects = {
                k: rotation_augmentation_fixed(o)
                for k, o in objects.items()
            }

        objects = {k: flip_augmentation(o) for k, o in objects.items()}

        objects = {k: jitter_augmentation(o) for k, o in objects.items()}

        objects = {k: np.ascontiguousarray(o) for k, o in objects.items()}

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}
        data['inputs']['scan_name'] = scannet_object_name
        data['inputs']['scan_content'] = objects['scan']
        data['inputs']['scan_mask'] = objects['mask']
        data['inputs']['cad_name'] = scannet_object_name
        data['inputs']['cad_content'] = objects['cad']
        data['inputs']['negative_name'] = scannet_object_name
        data['inputs']['negative_content'] = objects['negative']
        return data

    @staticmethod
    def _load_df(filepath):
        if os.path.splitext(filepath)[1] == ".mask":
            sample = load_mask(filepath)
            sample.tdf = 1.0 - sample.tdf.astype(np.float32)
        else:
            sample = load_raw_df(filepath)
        patch = sample.tdf
        return patch, sample

    def _load(self, path, mask_path=None):
        model, info = self._load_df(path)
        if self.mask_scans and mask_path is not None:
            mask_info = load_mask(mask_path)
            mask = mask_info.tdf
            info.tdf = np.where(mask, model, np.NINF)

        info = self.transformation(info)
        return info.tdf, info

    def __len__(self):
        return len(self.pairs)
