#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


def addBatch(data):
    for key in data['inputs'].keys():
        try:
            _ = data['inputs'][key].shape
            data['inputs'][key] = data['inputs'][key].unsqueeze(0)
        except:
            continue
    return data


def toDevice(data, device):
    for first_key in data.keys():
        for key, item in data[first_key].items():
            if isinstance(item, torch.Tensor):
                data[first_key][key] = data[first_key][key].to(
                    torch.device(device))
    return True


def toCuda(data):
    return toDevice(data, 'cuda')


def toCpu(data):
    return toDevice(data, 'cpu')


def toNumpy(data):
    for first_key in data.keys():
        for key, item in data[first_key].items():
            if isinstance(item, torch.Tensor):
                data[first_key][key] = data[first_key][key].detach().numpy()
    return True
