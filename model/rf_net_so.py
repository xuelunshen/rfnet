# -*- coding: utf-8 -*-
# @Time    : 2018-9-21 10:06
# @Author  : xylon
import torch

from model.rf_net_module import RFNetModule
from utils.net_utils import pair
from utils.image_utils import clip_patch, topk_map


class RFNetSO(RFNetModule):
    def __init__(self, det, des, SCORE_W, PAIR_W, PSIZE, TOPK):
        super(RFNetSO, self).__init__(det, des, SCORE_W, PAIR_W)

        self.PSIZE = PSIZE
        self.TOPK = TOPK

        self.det = det
        self.det.apply(self.det.weights_init)
        self.det.conv_o3.apply(self.det.convO_init)
        self.det.conv_o5.apply(self.det.convO_init)
        self.det.conv_o7.apply(self.det.convO_init)
        self.det.conv_o9.apply(self.det.convO_init)
        self.det.conv_o11.apply(self.det.convO_init)
        self.det.conv_o13.apply(self.det.convO_init)
        self.det.conv_o15.apply(self.det.convO_init)
        self.det.conv_o17.apply(self.det.convO_init)
        self.det.conv_o19.apply(self.det.convO_init)
        self.det.conv_o21.apply(self.det.convO_init)

        self.des = des
        self.des.apply(self.des.weights_init)

    def forward(self, batch):
        im1_data, im1_info, homo12, im2_data, im2_info, homo21, im1_raw, im2_raw = batch

        im1_rawsc, im1_scale, im1_orin = self.det(im1_data)
        im2_rawsc, im2_scale, im2_orin = self.det(im2_data)

        im1_gtscale, im1_gtorin = self.gt_scale_orin(
            im2_scale, im2_orin, homo12, homo21
        )
        im2_gtscale, im2_gtorin = self.gt_scale_orin(
            im1_scale, im1_orin, homo21, homo12
        )

        im1_gtsc, im1_topkmask, im1_topkvalue, im1_visiblemask = self.gtscore(
            im2_rawsc, homo12
        )
        im2_gtsc, im2_topkmask, im2_topkvalue, im2_visiblemask = self.gtscore(
            im1_rawsc, homo21
        )

        im1_score = self.det.process(im1_rawsc)[0]
        im2_score = self.det.process(im2_rawsc)[0]

        ###############################################################################
        # Extract patch and its descriptors by corresponding scale and orination
        ###############################################################################
        # (B*topk, 2, 32, 32)
        im1_ppair, im1_limc, im1_rimcw = pair(
            im1_topkmask,
            im1_topkvalue,
            im1_scale,
            im1_orin,
            im1_info,
            im1_raw,
            homo12,
            im2_gtscale,
            im2_gtorin,
            im2_info,
            im2_raw,
            self.PSIZE,
        )
        im2_ppair, im2_limc, im2_rimcw = pair(
            im2_topkmask,
            im2_topkvalue,
            im2_scale,
            im2_orin,
            im2_info,
            im2_raw,
            homo21,
            im1_gtscale,
            im1_gtorin,
            im1_info,
            im1_raw,
            self.PSIZE,
        )

        im1_lpatch, im1_rpatch = im1_ppair.chunk(chunks=2, dim=1)  # each is (N, 32, 32)
        im2_lpatch, im2_rpatch = im2_ppair.chunk(chunks=2, dim=1)  # each is (N, 32, 32)

        im1_lpdes, im1_rpdes = self.des(im1_lpatch), self.des(im1_rpatch)
        im2_lpdes, im2_rpdes = self.des(im2_lpatch), self.des(im2_rpatch)

        ###############################################################################
        # Extract patch and its descriptors by predicted scale and orination
        ###############################################################################
        # (B*topk, 2, 32, 32)
        im1_predpair, _, _ = pair(
            im1_topkmask,
            im1_topkvalue,
            im1_scale,
            im1_orin,
            im1_info,
            im1_raw,
            homo12,
            im2_scale,
            im2_orin,
            im2_info,
            im2_raw,
            self.PSIZE,
        )
        im2_predpair, _, _ = pair(
            im2_topkmask,
            im2_topkvalue,
            im2_scale,
            im2_orin,
            im2_info,
            im2_raw,
            homo21,
            im1_scale,
            im1_orin,
            im1_info,
            im1_raw,
            self.PSIZE,
        )

        # each is (N, 32, 32)
        im1_lpredpatch, im1_rpredpatch = im1_predpair.chunk(chunks=2, dim=1)
        im2_lpredpatch, im2_rpredpatch = im2_predpair.chunk(chunks=2, dim=1)

        im1_lpreddes, im1_rpreddes = self.des(im1_lpredpatch), self.des(im1_rpredpatch)
        im2_lpreddes, im2_rpreddes = self.des(im2_lpredpatch), self.des(im2_rpredpatch)

        endpoint = {
            "im1_score": im1_score,
            "im1_gtsc": im1_gtsc,
            "im1_visible": im1_visiblemask,
            "im2_score": im2_score,
            "im2_gtsc": im2_gtsc,
            "im2_visible": im2_visiblemask,
            "im1_limc": im1_limc,
            "im1_rimcw": im1_rimcw,
            "im2_limc": im2_limc,
            "im2_rimcw": im2_rimcw,
            "im1_lpdes": im1_lpdes,
            "im1_rpdes": im1_rpdes,
            "im2_lpdes": im2_lpdes,
            "im2_rpdes": im2_rpdes,
            "im1_lpreddes": im1_lpreddes,
            "im1_rpreddes": im1_rpreddes,
            "im2_lpreddes": im2_lpreddes,
            "im2_rpreddes": im2_rpreddes,
        }

        return endpoint

    def inference(self, im_data, im_info, im_raw):
        im_rawsc, im_scale, im_orint = self.det(im_data)
        im_score = self.det.process(im_rawsc)[0]
        im_topk = topk_map(im_score, self.TOPK)
        kpts = im_topk.nonzero()  # (B*topk, 4)
        cos, sim = im_orint.squeeze().chunk(chunks=2, dim=-1)
        cos = cos.masked_select(im_topk)  # (B*topk)
        sim = sim.masked_select(im_topk)  # (B*topk)
        im_orint = torch.cat((cos.unsqueeze(-1), sim.unsqueeze(-1)), dim=-1)
        im_patches = clip_patch(
            kpts,
            im_scale.masked_select(im_topk),
            im_orint,
            im_info,
            im_raw,
            PSIZE=self.PSIZE,
        )  # (numkp, 1, 32, 32)

        im_des = self.des(im_patches)

        return im_scale, kpts, im_des

    def detectAndCompute(self, im_path, device, output_size):
        """
        detect keypoints and compute its descriptor
        :param im_path: image path
        :param device: cuda or cpu
        :param output_size: resacle size
        :return: kp (#keypoints, 4) des (#keypoints, 128)
        """
        import numpy as np
        from skimage import io, color
        from utils.image_utils import im_rescale

        img = io.imread(im_path)

        # Gray
        img_raw = img = np.expand_dims(color.rgb2gray(img), -1)

        # Rescale
        # output_size = (240, 320)
        img, _, _, sw, sh = im_rescale(img, output_size)
        img_info = np.array([sh, sw])

        # to tensor
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = torch.from_numpy(img.transpose((2, 0, 1)))[None, :].to(
            device, dtype=torch.float
        )
        img_info = torch.from_numpy(img_info)[None, :].to(device, dtype=torch.float)
        img_raw = torch.from_numpy(img_raw.transpose((2, 0, 1)))[None, :].to(
            device, dtype=torch.float
        )

        # inference
        _, kp, des = self.inference(img, img_info, img_raw)

        return kp, des, img
