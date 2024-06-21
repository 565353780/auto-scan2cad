# -*- coding: utf-8 -*-
# @Author: Thibault GROUEIX
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-18 15:06:25
# @Email:  cshzxie@gmail.com

import chamfer
import torch


class ChamferFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)

        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2,
                                                grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


class ChamferDistanceL2(torch.nn.Module):
    f''' Chamder Distance L2
    '''

    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1) + torch.mean(dist2)


class ChamferDistanceL2_split(torch.nn.Module):
    f''' Chamder Distance L2
    '''

    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return torch.mean(dist1), torch.mean(dist2)


class ChamferDistanceL1(torch.nn.Module):
    f''' Chamder Distance L1
    '''

    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(xyz1, dim=2).ne(0)
            non_zeros2 = torch.sum(xyz2, dim=2).ne(0)
            xyz1 = xyz1[non_zeros1].unsqueeze(dim=0)
            xyz2 = xyz2[non_zeros2].unsqueeze(dim=0)

        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        # import pdb
        # pdb.set_trace()
        dist1 = torch.sqrt(dist1)
        dist2 = torch.sqrt(dist2)
        return (torch.mean(dist1) + torch.mean(dist2)) / 2


class WeightedChamferDistanceL2(torch.nn.Module):
    f''' Weighted Chamder Distance L2
    '''

    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, partial, infer, complete):
        batch_size = partial.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(partial, dim=2).ne(0)
            non_zeros2 = torch.sum(infer, dim=2).ne(0)
            non_zeros3 = torch.sum(complete, dim=2).ne(0)
            partial = partial[non_zeros1].unsqueeze(dim=0)
            infer = infer[non_zeros2].unsqueeze(dim=0)
            complete = complete[non_zeros3].unsqueeze(dim=0)

        _, dist_cp = ChamferFunction.apply(partial, complete)
        dist_ic, dist_ci = ChamferFunction.apply(infer, complete)
        max_dist_cp = torch.max(dist_cp)
        weighted_dist_ci = dist_cp * dist_ci / max_dist_cp
        return torch.mean(dist_ic) + torch.mean(weighted_dist_ci)


class WeightedChamferDistanceL1(torch.nn.Module):
    f''' Weighted Chamder Distance L1
    '''

    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, partial, infer, complete):
        batch_size = partial.size(0)
        if batch_size == 1 and self.ignore_zeros:
            non_zeros1 = torch.sum(partial, dim=2).ne(0)
            non_zeros2 = torch.sum(infer, dim=2).ne(0)
            non_zeros3 = torch.sum(complete, dim=2).ne(0)
            partial = partial[non_zeros1].unsqueeze(dim=0)
            infer = infer[non_zeros2].unsqueeze(dim=0)
            complete = complete[non_zeros3].unsqueeze(dim=0)

        _, dist_cp = ChamferFunction.apply(partial, complete)
        dist_ic, dist_ci = ChamferFunction.apply(infer, complete)
        max_dist_cp = torch.max(dist_cp)
        weighted_dist_ci = dist_cp * dist_ci / max_dist_cp

        dist1 = torch.sqrt(dist_ic)
        dist2 = torch.sqrt(weighted_dist_ci)
        return (torch.mean(dist1) + torch.mean(dist2)) / 2
