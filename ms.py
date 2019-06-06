# -*- coding: utf-8 -*-
# @Time    : 2018-10-10 15:01
# @Author  : xylon
import random
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader

from utils.train_utils import parse_batch

parser = argparse.ArgumentParser(description="Test")
parser.add_argument("--ct", default=5.0, type=float)  # pixel distance threshold
parser.add_argument("--data", default="v", type=str)  # data sequence
parser.add_argument("--resume", default=None, type=str)  # model path
args = parser.parse_args()


if __name__ == "__main__":
    from hpatch_dataset import *
    from utils.eval_utils import *
    from utils.common_utils import gct

    from model.rf_des import HardNetNeiMask
    from model.rf_det_so import RFDetSO
    from model.rf_net_so import RFNetSO
    from config import cfg

    print(f"{gct()} : start time")

    cfg.MODEL.COO_THRSH = args.ct

    random.seed(cfg.PROJ.SEED)
    torch.manual_seed(cfg.PROJ.SEED)
    np.random.seed(cfg.PROJ.SEED)

    ###############################################################################
    # Load test data
    ###############################################################################
    PPT = [cfg.PROJ.TRAIN_PPT, (cfg.PROJ.TRAIN_PPT + cfg.PROJ.EVAL_PPT)]
    use_all = {"view": False, "illu": True, "ef": True}
    print(f"{gct()} : load test data")
    data_loader = DataLoader(
        HpatchDataset(
            data_type="test",
            PPT=PPT,
            use_all=use_all[args.data],
            csv_file=cfg[args.data]["csv"],
            root_dir=cfg[args.data]["root"],
            transform=transforms.Compose(
                [
                    Grayscale(),
                    Normalize(mean=cfg[args.data]["MEAN"], std=cfg[args.data]["STD"]),
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

    print(f"{gct()} : model init")
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

    print(f"{gct()} : to device")
    device = torch.device("cuda")
    model = model.to(device)
    resume = args.resume
    print(f"{gct()} : in {resume}")
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint["state_dict"])

    print(f"{gct()} : start eval")
    model.eval()
    PreNN, PreNNT, PreNNDR = 0, 0, 0
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_loader, 1):
            batch = parse_batch(sample_batched, device)

            TPNN, PNN, TPNNT, PNNT, TPNNDR, PNNDR = eval_model(
                model, batch, cfg.MODEL.DES_THRSH, cfg.MODEL.COO_THRSH
            )

            PreNN += TPNN / PNN
            PreNNT += TPNNT / PNNT
            PreNNDR += TPNNDR / PNNDR

    length = len(data_loader)
    PreNN, PreNNT, PreNNDR = (PreNN / length, PreNNT / length, PreNNDR / length)
    meanms = (PreNN + PreNNT + PreNNDR) / 3
    checkpoint_name = (
        f"data: {args.data} "
        f"len: {len(data_loader):3d} "
        f"ct: {args.ct}\t"
        f"NN: {PreNN:.3f} "
        f"NNT: {PreNNT:.3f} "
        f"NNR: {PreNNDR:.3f} "
        f"mean: {meanms:.3f}"
    )
    print(checkpoint_name)
    print(f"{gct()} : finish eval")
    print(f"\n")
