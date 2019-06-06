# -*- coding: utf-8 -*-
# @Time    : 2018-7-27 10:48
# @Author  : xylon
import os
import time
import torch
import random
import argparse
import numpy as np
from torch import autograd
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from hpatch_dataset import (
    HpatchDataset,
    Grayscale,
    Normalize,
    Rescale,
    LargerRescale,
    RandomCrop,
    ToTensor,
)
from utils.eval_utils import eval_model, getAC
from utils.common_utils import gct, prettydict
from utils.train_utils import (
    parse_batch,
    parse_unsqueeze,
    mgpu_merge,
    writer_log,
    ExponentialLR,
    SgdLR,
)


def Lr_Schechuler(lr_schedule, optimizer, epoch, cfg):
    if lr_schedule == "exp":
        ExponentialLR(optimizer, epoch, cfg)
    elif lr_schedule == "sgd":
        SgdLR(optimizer, cfg)


def select_optimizer(optim, param, lr, wd):
    if optim == "sgd":
        optimizer = torch.optim.SGD(
            param, lr=lr, momentum=0.9, dampening=0.9, weight_decay=wd
        )
    elif optim == "adam":
        optimizer = torch.optim.Adam(param, lr=lr, weight_decay=wd)
    else:
        raise Exception(f"Not supported optimizer: {optim}")
    return optimizer


def create_optimizer(
    det_optim, des_optim, model, det_lr, des_lr, det_wd, des_wd, mgpu=False
):
    if mgpu:
        det_param = model.module.det.parameters()
        des_param = model.module.des.parameters()
    else:
        det_param = model.det.parameters()
        des_param = model.des.parameters()

    det_optimizer = select_optimizer(det_optim, det_param, det_lr, det_wd)
    des_optimizer = select_optimizer(des_optim, des_param, des_lr, des_wd)

    return det_optimizer, des_optimizer


def parse_parms():
    parser = argparse.ArgumentParser(description="Test a DualDet Network")
    parser.add_argument(
        "--resume", default="", type=str, help="latest checkpoint (default: none)"
    )
    parser.add_argument(
        "--ver", default="", type=str, help="model version(defualt: none)"
    )
    parser.add_argument(
        "--save", default="", type=str, help="source code save path(defualt: none)"
    )
    parser.add_argument(
        "--det-step", default=1, type=int, help="train detection step(defualt: 1)"
    )
    parser.add_argument(
        "--des-step", default=2, type=int, help="train descriptor step(defualt: 2)"
    )
    return parser.parse_args()


def reserve_mem():
    try:
        gpuid = int(os.environ["CUDA_VISIBLE_DEVICES"])
    except:
        gpuid = -1

    smi = (
        os.popen(
            "nvidia-smi --query-gpu=memory.total,memory.used --format=csv,nounits,noheader"
        )
        .read()
        .strip()
        .replace("\n", ",")
        .replace(" ", "")
        .split(",")
    )
    total, used = int(smi[2 * gpuid]), int(smi[2 * gpuid + 1])
    block_ratio = 0.90
    max_mem = int(total * block_ratio)
    block_mem = max_mem - used
    x = torch.rand((256, 1024, block_mem)).cuda()
    x = torch.rand((2, 2)).cuda()
    print(
        f"{gct()} : GPU usage total:{total} used:{used} max_mem:{max_mem} block_ratio:{block_ratio} block_mem:{block_mem}"
    )


if __name__ == "__main__":
    from config import cfg
    from model.rf_det_so import RFDetSO
    from model.rf_des import HardNetNeiMask
    from model.rf_net_so import RFNetSO

    # reserving gpu memory
    # print(f"{gct()} : Reserving memory")
    # reserve_mem()

    args = parse_parms()
    cfg.TRAIN.SAVE = args.save
    cfg.TRAIN.DET = args.det_step
    cfg.TRAIN.DES = args.des_step
    print(f"{gct()} : Called with args:{args}")
    print(f"{gct()} : Using config:")
    prettydict(cfg)

    ###############################################################################
    # Set the random seed manually for reproducibility
    ###############################################################################
    print(f"{gct()} : Prepare for repetition")
    device = torch.device("cuda" if cfg.PROJ.USE_GPU else "cpu")
    mgpu = True if cfg.PROJ.USE_GPU and torch.cuda.device_count() > 1 else False
    seed = cfg.PROJ.SEED
    if cfg.PROJ.USE_GPU:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
        if mgpu:
            print(f"{gct()} : Train with {torch.cuda.device_count()} GPUs")
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    ###############################################################################
    # Build the model
    ###############################################################################
    print(f"{gct()} : Build the model")
    det = RFDetSO(
        cfg.TRAIN.score_com_strength,
        cfg.TRAIN.scale_com_strength,
        cfg.TRAIN.NMS_THRESH,
        cfg.TRAIN.NMS_KSIZE,
        cfg.TRAIN.TOPK,
        cfg.MODEL.GAUSSIAN_KSIZE,
        cfg.MODEL.GAUSSIAN_SIGMA,
        cfg.MODEL.KSIZE,
        cfg.MODEL.padding,
        cfg.MODEL.dilation,
        cfg.MODEL.scale_list,
    )
    des = HardNetNeiMask(cfg.HARDNET.MARGIN, cfg.MODEL.COO_THRSH)
    model = RFNetSO(
        det, des, cfg.LOSS.SCORE, cfg.LOSS.PAIR, cfg.PATCH.SIZE, cfg.TRAIN.TOPK
    )
    if mgpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device=device)

    ###############################################################################
    # Load train data
    ###############################################################################
    PPT = [cfg.PROJ.TRAIN_PPT, (cfg.PROJ.TRAIN_PPT + cfg.PROJ.EVAL_PPT)]

    print(f"{gct()} : Loading traning data")
    train_data = DataLoader(
        HpatchDataset(
            data_type="train",
            PPT=PPT,
            use_all=cfg.PROJ.TRAIN_ALL,
            csv_file=cfg[cfg.PROJ.TRAIN]["csv"],
            root_dir=cfg[cfg.PROJ.TRAIN]["root"],
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(
                        mean=cfg[cfg.PROJ.TRAIN]["MEAN"], std=cfg[cfg.PROJ.TRAIN]["STD"]
                    ),
                    LargerRescale((960, 1280)),
                    RandomCrop((720, 960)),
                    Rescale((240, 320)),
                    ToTensor(),
                ]
            ),
        ),
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    ###############################################################################
    # Load evaluation data
    ###############################################################################
    print(f"{gct()} : Loading evaluation data")
    val_data = DataLoader(
        HpatchDataset(
            data_type="eval",
            PPT=PPT,
            use_all=cfg.PROJ.EVAL_ALL,
            csv_file=cfg[cfg.PROJ.EVAL]["csv"],
            root_dir=cfg[cfg.PROJ.EVAL]["root"],
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(
                        mean=cfg[cfg.PROJ.EVAL]["MEAN"], std=cfg[cfg.PROJ.EVAL]["STD"]
                    ),
                    Rescale((960, 1280)),
                    Rescale((240, 320)),
                    ToTensor(),
                ]
            ),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    ###############################################################################
    # Load test data
    ###############################################################################
    print(f"{gct()} : Loading testing data")
    test_data = DataLoader(
        HpatchDataset(
            data_type="test",
            PPT=PPT,
            use_all=cfg.PROJ.TEST_ALL,
            csv_file=cfg[cfg.PROJ.TEST]["csv"],
            root_dir=cfg[cfg.PROJ.TEST]["root"],
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(
                        mean=cfg[cfg.PROJ.TEST]["MEAN"], std=cfg[cfg.PROJ.TEST]["STD"]
                    ),
                    Rescale((960, 1280)),
                    Rescale((240, 320)),
                    ToTensor(),
                ]
            ),
        ),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    ###############################################################################
    # Build the optimizer
    ###############################################################################
    det_optim, des_optim = create_optimizer(
        det_optim=cfg.TRAIN.DET_OPTIMIZER,
        des_optim=cfg.TRAIN.DES_OPTIMIZER,
        model=model,
        det_lr=cfg.TRAIN.DET_LR,
        des_lr=cfg.TRAIN.DES_LR,
        det_wd=cfg.TRAIN.DET_WD,
        des_wd=cfg.TRAIN.DES_WD,
        mgpu=mgpu,
    )

    ###############################################################################
    # resume model if exists
    ###############################################################################
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"{gct()} : Loading checkpoint {args.resume}")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            det_optim.load_state_dict(checkpoint["det_optim"])
            des_optim.load_state_dict(checkpoint["des_optim"])
        else:
            print(f"{gct()} : Cannot found checkpoint {args.resume}")
    else:
        args.start_epoch = 0

    ###############################################################################
    # Visualization
    ###############################################################################
    train_writer = SummaryWriter(f"{args.save}/log/train")
    test_writer = SummaryWriter(f"{args.save}/log/test")

    ###############################################################################
    # Training function
    ###############################################################################
    def train():
        start_time = time.time()
        for i_batch, sample_batched in enumerate(train_data, 1):
            model.train()
            batch = parse_batch(sample_batched, device)
            with autograd.detect_anomaly():
                for des_train in range(0, cfg.TRAIN.DES):
                    model.zero_grad()
                    des_optim.zero_grad()
                    endpoint = model(batch)
                    _, _, desloss = (
                        model.module.criterion(endpoint)
                        if mgpu
                        else model.criterion(endpoint)
                    )
                    desloss.backward()
                    des_optim.step()
                for det_train in range(0, cfg.TRAIN.DET):
                    model.zero_grad()
                    det_optim.zero_grad()
                    endpoint = model(batch)
                    _, detloss, _ = (
                        model.module.criterion(endpoint)
                        if mgpu
                        else model.criterion(endpoint)
                    )
                    detloss.backward()
                    det_optim.step()

            Lr_Schechuler(cfg.TRAIN.DET_LR_SCHEDULE, det_optim, epoch, cfg)
            Lr_Schechuler(cfg.TRAIN.DES_LR_SCHEDULE, des_optim, epoch, cfg)

            # log
            if i_batch % cfg.TRAIN.LOG_INTERVAL == 0 and i_batch > 0:
                elapsed = time.time() - start_time
                model.eval()
                with torch.no_grad():
                    eptr = model(parse_unsqueeze(train_data.dataset[0], device))
                    PLT, cur_detloss, cur_desloss = (
                        model.module.criterion(eptr) if mgpu else model.criterion(eptr)
                    )

                    PLTS = PLT["scalar"]
                    PLTS["Accuracy"] = getAC(eptr["im1_lpdes"], eptr["im1_rpdes"])
                    PLTS["det_lr"] = det_optim.param_groups[0]["lr"]
                    PLTS["des_lr"] = des_optim.param_groups[0]["lr"]
                    if mgpu:
                        mgpu_merge(PLTS)
                    iteration = (epoch - 1) * len(train_data) + (i_batch - 1)
                    writer_log(train_writer, PLT["scalar"], iteration)

                    pstring = (
                        "epoch {:2d} | {:4d}/{:4d} batches | ms {:4.02f} | "
                        "sco {:07.05f} | pair {:05.03f} | des {:05.03f} |".format(
                            epoch,
                            i_batch,
                            len(train_data) // cfg.TRAIN.BATCH_SIZE,
                            elapsed / cfg.TRAIN.LOG_INTERVAL,
                            PLTS["score_loss"],
                            PLTS["pair_loss"],
                            PLTS["hard_loss"],
                        )
                    )

                    # eval log
                    # parsed_valbatch = parse_unsqueeze(val_data.dataset[0], device)
                    # ept = model(parsed_valbatch)
                    ept = model(parse_unsqueeze(val_data.dataset[0], device))
                    PLT, _, _ = (
                        model.module.criterion(ept) if mgpu else model.criterion(ept)
                    )
                    PLTS = PLT["scalar"]
                    PLTS["Accuracy"] = getAC(ept["im1_lpdes"], ept["im1_rpdes"])
                    writer_log(test_writer, PLT["scalar"], iteration)

                    print(f"{gct()} | {pstring}")
                    start_time = time.time()

    ###############################################################################
    # evaluate function
    ###############################################################################
    def evaluate(data_source):
        model.eval()
        PreNN, PreNNT, PreNNDR = 0, 0, 0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(data_source, 1):
                batch = parse_batch(sample_batched, device)

                TPNN, PNN, TPNNT, PNNT, TPNNDR, PNNDR = eval_model(
                    model.module if mgpu else model,
                    batch,
                    cfg.MODEL.DES_THRSH,
                    cfg.MODEL.COO_THRSH,
                )

                PreNN += TPNN / PNN
                PreNNT += TPNNT / PNNT
                PreNNDR += TPNNDR / PNNDR

        length = len(data_source)
        PreNN, PreNNT, PreNNDR = (PreNN / length, PreNNT / length, PreNNDR / length)
        meanms = (PreNN + PreNNT + PreNNDR) / 3
        checkpoint_name = (
            f"NN_{PreNN:.3f}_NNT_{PreNNT:.3f}_NNDR_{PreNNDR:.3f}_MeanMS_{meanms:.3f}"
        )
        return checkpoint_name, meanms

    ###############################################################################
    # Training code
    ###############################################################################
    print(f"{gct()} : Start training")
    best_ms = None
    best_f = None
    start_epoch = args.start_epoch + 1
    end = cfg.TRAIN.EPOCH_NUM
    for epoch in range(start_epoch, end):
        epoch_start_time = time.time()
        train()
        checkpoint, val_ms = evaluate(val_data)

        # Save the model if the match score is the best we've seen so far.
        if not best_ms or val_ms >= best_ms:
            state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "det_optim": det_optim.state_dict(),
                "des_optim": des_optim.state_dict(),
            }
            filename = f"{args.save}/model/e{epoch:03d}_{checkpoint}.pth.tar"
            torch.save(state, filename)
            best_ms = val_ms
            best_f = filename

        print("-" * 96)
        print(
            "| end of epoch {:3d} | time: {:5.02f}s | val ms {:5.03f} | best ms {:5.03f} | ".format(
                epoch, (time.time() - epoch_start_time), val_ms, best_ms
            )
        )
        print("-" * 96)

    # Load the best saved model.
    with open(best_f, "rb") as f:
        model.load_state_dict(torch.load(f)["state_dict"])

    # Run on test data.
    _, test_ms = evaluate(test_data)
    print("=" * 96)
    print("| End of training | test ms {:5.03f}".format(test_ms))
    print("=" * 96)
