#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


def cutLoss(data, loss_name, min_value=None, max_value=None):
    if min_value is not None:
        data['losses'][loss_name] = torch.max(
            data['losses'][loss_name],
            torch.tensor(min_value).to(torch.float32).to(
                data['losses'][loss_name].device).reshape(1))[0]

    if max_value is not None:
        data['losses'][loss_name] = torch.min(
            data['losses'][loss_name],
            torch.tensor(max_value).to(torch.float32).to(
                data['losses'][loss_name].device).reshape(1))[0]
    return data


def setWeight(data, loss_name, weight, min_value=None, max_value=None):
    if weight != 1:
        data['losses'][loss_name] = data['losses'][loss_name] * weight

    data = cutLoss(data, loss_name, min_value, max_value)
    return data
