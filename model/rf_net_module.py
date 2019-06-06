# -*- coding: utf-8 -*-
# @Time    : 2018-9-27 19:42
# @Author  : xylon

import torch
import torch.nn as nn

from utils.common_utils import imgBatchXYZ, transXYZ_2_to_1
from utils.image_utils import warp, filter_border
from utils.math_utils import MSD, distance_matrix_vector, L2Norm


class RFNetModule(nn.Module):
    def __init__(self, det, des, SCORE_W, PAIR_W):
        super(RFNetModule, self).__init__()

        self.det = det
        self.des = des
        self.SCORE_W = SCORE_W
        self.PAIR_W = PAIR_W

    def forward(self, **kwargs):
        pass

    def inference(self, **kwargs):
        pass

    def gtscore(self, right_score, homolr):
        im2_score = right_score
        im2_score = filter_border(im2_score)

        # warp im2_score to im1w_score and calculate visible_mask
        im1w_score = warp(im2_score, homolr)
        im1visible_mask = warp(
            im2_score.new_full(im2_score.size(), fill_value=1, requires_grad=True),
            homolr,
        )

        im1gt_score, topk_mask, topk_value = self.det.process(im1w_score)

        return im1gt_score, topk_mask, topk_value, im1visible_mask

    @staticmethod
    def gt_scale_orin(im2_scale, im2_orin, homo12, homo21):
        B, H, W, C = im2_scale.size()
        im2_cos, im2_sin = im2_orin.squeeze().chunk(chunks=2, dim=-1)  # (B, H, W, 1)
        # im2_tan = im2_sin / im2_cos

        # each centX, centY, centZ is (B, H, W, 1)
        centX, centY, centZ = imgBatchXYZ(B, H, W).to(im2_scale.device).chunk(3, dim=3)

        """get im1w scale maps"""
        half_scale = im2_scale // 2
        centXYZ = torch.cat((centX, centY, centZ), dim=3)  # (B, H, W, 3)
        upXYZ = torch.cat((centX, centY - half_scale, centZ), dim=3)
        bottomXYZ = torch.cat((centX, centY + half_scale, centZ), dim=3)
        rightXYZ = torch.cat((centX + half_scale, centY, centZ), dim=3)
        leftXYZ = torch.cat((centX - half_scale, centY, centZ), dim=3)

        centXYw = transXYZ_2_to_1(centXYZ, homo21)  # (B, H, W, 2) (x, y)
        centXw, centYw = centXYw.chunk(chunks=2, dim=-1)  # (B, H, W, 1)
        centXYw = centXYw.long()
        upXYw = transXYZ_2_to_1(upXYZ, homo21).long()
        rightXYw = transXYZ_2_to_1(rightXYZ, homo21).long()
        bottomXYw = transXYZ_2_to_1(bottomXYZ, homo21).long()
        leftXYw = transXYZ_2_to_1(leftXYZ, homo21).long()

        upScale = MSD(upXYw, centXYw)
        rightScale = MSD(rightXYw, centXYw)
        bottomScale = MSD(bottomXYw, centXYw)
        leftScale = MSD(leftXYw, centXYw)
        centScale = (upScale + rightScale + bottomScale + leftScale) / 4  # (B, H， W, 1)

        """get im1w orintation maps"""
        offset_x, offset_y = im2_scale * im2_cos, im2_scale * im2_sin  # (B, H, W, 1)
        offsetXYZ = torch.cat((centX + offset_x, centY + offset_y, centZ), dim=3)
        offsetXYw = transXYZ_2_to_1(offsetXYZ, homo21)  # (B, H, W, 2) (x, y)
        offsetXw, offsetYw = offsetXYw.chunk(chunks=2, dim=-1)  # (B, H, W, 1)
        offset_ww, offset_hw = offsetXw - centXw, offsetYw - centYw  # (B, H, W, 1)
        offset_rw = (offset_ww ** 2 + offset_hw ** 2 + 1e-8).sqrt()
        # tan = offset_hw / (offset_ww + 1e-8)  # (B, H, W, 1)
        cos_w = offset_ww / (offset_rw + 1e-8)  # (B, H, W, 1)
        sin_w = offset_hw / (offset_rw + 1e-8)  # (B, H, W, 1)
        # atan_w = np.arctan(tan.cpu().detach())  # (B, H, W, 1)

        # get left scale by transXYZ_2_to_1
        map_xy_2_to_1 = transXYZ_2_to_1(centXYZ, homo12).round().long()  # (B, H, W, 2)
        x, y = map_xy_2_to_1.chunk(2, dim=3)  # each x and y is (B, H, W, 1)
        x = x.clamp(min=0, max=W - 1)
        y = y.clamp(min=0, max=H - 1)

        # (B, H, W, 1)
        im1w_scale = centScale[
            torch.arange(B)[:, None].repeat(1, H * W), y.view(B, -1), x.view(B, -1)
        ].view(im2_scale.size())

        # (B, H, W, 1, 2)
        im1w_cos = cos_w[
            torch.arange(B)[:, None].repeat(1, H * W), y.view(B, -1), x.view(B, -1)
        ].view(im2_cos.size())
        im1w_sin = sin_w[
            torch.arange(B)[:, None].repeat(1, H * W), y.view(B, -1), x.view(B, -1)
        ].view(im2_sin.size())
        im1w_orin = torch.cat((im1w_cos[:, None], im1w_sin[:, None]), dim=-1)
        im1w_orin = L2Norm(im1w_orin, dim=-1).to(im2_orin.device)

        return im1w_scale, im1w_orin

    def criterion(self, endpoint):

        im1_score = endpoint["im1_score"]
        im1_gtsc = endpoint["im1_gtsc"]
        im1_visible = endpoint["im1_visible"]

        im2_score = endpoint["im2_score"]
        im2_gtsc = endpoint["im2_gtsc"]
        im2_visible = endpoint["im2_visible"]

        im1_limc = endpoint["im1_limc"]
        im1_rimcw = endpoint["im1_rimcw"]
        im2_limc = endpoint["im2_limc"]
        im2_rimcw = endpoint["im2_rimcw"]

        im1_lpdes = endpoint["im1_lpdes"]
        im1_rpdes = endpoint["im1_rpdes"]
        im2_lpdes = endpoint["im2_lpdes"]
        im2_rpdes = endpoint["im2_rpdes"]

        im1_lpreddes = endpoint["im1_lpreddes"]
        im1_rpreddes = endpoint["im1_rpreddes"]
        im2_lpreddes = endpoint["im2_lpreddes"]
        im2_rpreddes = endpoint["im2_rpreddes"]

        #
        # score loss
        #
        im1_scloss = self.det.loss(im1_score, im1_gtsc, im1_visible)
        im2_scloss = self.det.loss(im2_score, im2_gtsc, im2_visible)
        score_loss = (im1_scloss + im2_scloss) / 2.0 * self.SCORE_W

        #
        # pair loss
        #
        im1_pairloss = distance_matrix_vector(im1_lpreddes, im1_rpreddes).diag().mean()
        im2_pairloss = distance_matrix_vector(im2_lpreddes, im2_rpreddes).diag().mean()
        pair_loss = (im1_pairloss + im2_pairloss) / 2.0 * self.PAIR_W

        #
        # hard loss
        #
        im1_hardloss = self.des.loss(im1_lpdes, im1_rpdes, im1_limc, im1_rimcw)
        im2_hardloss = self.des.loss(im2_lpdes, im2_rpdes, im2_limc, im2_rimcw)
        hard_loss = (im1_hardloss + im2_hardloss) / 2.0

        # loss summary
        det_loss = score_loss + pair_loss
        des_loss = hard_loss

        PLT_SCALAR = {}
        PLT = {"scalar": PLT_SCALAR}

        PLT_SCALAR["score_loss"] = score_loss
        PLT_SCALAR["pair_loss"] = pair_loss
        PLT_SCALAR["hard_loss"] = hard_loss

        return PLT, det_loss.mean(), des_loss.mean()
