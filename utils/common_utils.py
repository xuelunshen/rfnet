# -*- coding: utf-8 -*-
# @Time    : 2018-8-2 9:20
# @Author  : xylon
import time
import torch


def gct(f="l"):
    """
    get current time
    :param f: "l" for log, "f" for file name
    :return: formatted time
    """
    if f == "l":
        return time.strftime("%m-%d %H:%M:%S", time.localtime(time.time()))
    elif f == "f":
        return f'{time.strftime("%m_%d_%H_%M", time.localtime(time.time()))}'


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def prettydict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("\t" * indent + f"{key}")
            prettydict(value, indent + 1)
        else:
            print("\t" * indent + f"{key:>18} : {value}")


def unsqueezebatch(batch):
    for key in batch:
        batch[key] = batch[key].unsqueeze(0)
    return batch


def isnan(t):
    return torch.isnan(t).sum().item() > 0


def imgBatchXYZ(B, H, W):
    Ha, Wa = torch.arange(H), torch.arange(W)
    gy, gx = torch.meshgrid([Ha, Wa])
    gx, gy = torch.unsqueeze(gx.float(), -1), torch.unsqueeze(gy.float(), -1)
    ones = gy.new_full(gy.size(), fill_value=1)
    grid = torch.cat((gx, gy, ones), -1)  # (H, W, 3)
    grid = torch.unsqueeze(grid, 0)  # (1, H, W, 3)
    grid = grid.repeat(B, 1, 1, 1)  # (B, H, W, 3)
    return grid


def transXYZ_2_to_1(batchXYZ, homo21):
    """
    project each pixel in right to left xy coordination
    :param batchXYZ: (B, H, W, 3)
    :param homo21: (B, 3, 3)
    :return: warp XYZ: (B, H, W, 2)
    """
    B, H, W, C = batchXYZ.size()
    grid = batchXYZ.contiguous().view(B, H * W, C)  # (B, H*W, 3)
    grid = grid.contiguous().permute(0, 2, 1)  # (B, 3, H*W)
    grid = grid.type_as(homo21).to(homo21.device)

    grid_w = torch.matmul(homo21, grid)  # (B, 3, 3) matmul (B, 3, H*W) => (B, 3, H*W)
    grid_w = grid_w.contiguous().permute(0, 2, 1)  # (B, H*W, 3)
    grid_w = torch.div(
        grid_w, torch.unsqueeze(grid_w[:, :, 2], -1) + 1e-8
    )  # (B, H*W, 3)
    grid_w = grid_w.contiguous().view(B, H, W, -1)[:, :, :, :2]  # (B, H, W, 2)

    return grid_w
