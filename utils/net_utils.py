# -*- coding: utf-8 -*-
# @Time    : 2018-9-28 15:58
# @Author  : xylon
import torch

from utils.math_utils import ptCltoCr
from utils.image_utils import clip_patch


def pair(
    left_imtopk,
    left_topkvalue,
    left_imscale,
    left_imorint,
    left_iminfo,
    left_imraw,
    homolr,
    right_imscale,
    right_imorint,
    right_iminfo,
    right_imraw,
    PSIZE,
):
    """
    generate patch pair based on left topk_mask
    """
    left_imC = left_imtopk.nonzero()  # (B*topk, 4)
    left_imS = left_imscale.masked_select(
        left_imtopk
    )  # (B*topk) have grad to detect network
    if left_imorint is not None:
        left_cos, left_sim = left_imorint.squeeze().chunk(chunks=2, dim=-1)
        left_cos = left_cos.masked_select(left_imtopk)  # (B*topk)
        left_sim = left_sim.masked_select(left_imtopk)  # (B*topk)
        left_imO = torch.cat((left_cos.unsqueeze(-1), left_sim.unsqueeze(-1)), dim=-1)
    else:
        left_imO = None

    left_imP = clip_patch(
        left_imC, left_imS, left_imO, left_iminfo, left_imraw, PSIZE=PSIZE
    )  # (numkp, 1, 32, 32)

    right_imC, right_imS, right_imO = ptCltoCr(
        left_imC, homolr, right_imscale, right_imorint
    )
    right_imP = clip_patch(
        right_imC, right_imS, right_imO, right_iminfo, right_imraw, PSIZE=PSIZE
    )  # (numkp, 1, 32, 32)

    left_impair = torch.cat((left_imP, right_imP), 1)  # (numkp, 2, 32, 32)

    return left_impair, left_imC, right_imC
