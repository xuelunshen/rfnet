# -*- coding: utf-8 -*-
# @Time    : 2018-9-21 14:32
# @Author  : xylon
import numpy as np
import torch


def parse_batch(batch, device):
    im1_data = batch["im1"].to(device, dtype=torch.float)
    im1_info = batch["im1_info"].to(device, dtype=torch.float)
    homo12 = batch["homo12"].to(device, dtype=torch.float)
    im2_data = batch["im2"].to(device, dtype=torch.float)
    im2_info = batch["im2_info"].to(device, dtype=torch.float)
    homo21 = batch["homo21"].to(device, dtype=torch.float)
    im1_raw = batch["im1_raw"].to(device, dtype=torch.float)
    im2_raw = batch["im2_raw"].to(device, dtype=torch.float)
    return im1_data, im1_info, homo12, im2_data, im2_info, homo21, im1_raw, im2_raw


def parse_unsqueeze(batch, device):
    im1_data = batch["im1"].to(device, dtype=torch.float)
    im1_info = batch["im1_info"].to(device, dtype=torch.float)
    homo12 = batch["homo12"].to(device, dtype=torch.float)
    im2_data = batch["im2"].to(device, dtype=torch.float)
    im2_info = batch["im2_info"].to(device, dtype=torch.float)
    homo21 = batch["homo21"].to(device, dtype=torch.float)
    im1_raw = batch["im1_raw"].to(device, dtype=torch.float)
    im2_raw = batch["im2_raw"].to(device, dtype=torch.float)
    return (
        im1_data.unsqueeze(0),
        im1_info.unsqueeze(0),
        homo12.unsqueeze(0),
        im2_data.unsqueeze(0),
        im2_info.unsqueeze(0),
        homo21.unsqueeze(0),
        im1_raw.unsqueeze(0),
        im2_raw.unsqueeze(0),
    )


def writer_log(writer, PLT_SCALAR, iteration):
    for key in PLT_SCALAR:
        writer.add_scalar(f"data/{key}", PLT_SCALAR[key], iteration)


def mgpu_merge(PLT_SCALAR):
    for key in PLT_SCALAR:
        PLT_SCALAR[key] = PLT_SCALAR[key].mean()


def netparam_log(writer, model, iteration):
    for tag, value in model.named_parameters():
        tag = tag.replace(".", "/")
        writer.add_histogram(tag, value, iteration)
        writer.add_histogram(tag + "/grad", value.grad, iteration)


def ExponentialLR(optimizer, epoch, cfg):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    decay_epoch = cfg.TRAIN.LR_DECAY_EPOCH
    lr = cfg.TRAIN.DET_LR
    lr_baseline = cfg.TRAIN.LR_BASE

    if epoch % decay_epoch:
        return

    new_lr = lr * (0.9 ** (epoch // decay_epoch))
    new_lr = max(new_lr, lr_baseline)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def SgdLR(optimizer, cfg):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if "step" not in group:
            group["step"] = 0.0
        else:
            group["step"] += 1.0
        group["lr"] = cfg.TRAIN.DES_LR * (
            1.0
            - float(group["step"])
            * float(cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.TOPK)
            / float(
                np.round(cfg.PROJ.TRAIN_PPT * cfg[cfg.PROJ.TRAIN]["NUM"])
                * cfg.TRAIN.TOPK
                * cfg.TRAIN.EPOCH_NUM
            )
        )
