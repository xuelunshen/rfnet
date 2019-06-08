# -*- coding: utf-8 -*-
# @Time    : 2018-8-4 14:53
# @Author  : xylon

import os
import torch
import numpy as np
import pandas as pd
from skimage import io, transform, color
from torch.utils.data import Dataset

from utils.common_utils import gct
from utils.image_utils import im_rescale


class HpatchDataset(Dataset):
    """HPatch dataset."""

    def __init__(
        self, csv_file, root_dir, data_type, use_all, PPT, transform=None, skiprows=None
    ):
        """
        Args:
            csv_file (string):  Path to the csv file with annotations.
            root_dir (string):  Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        """ folder,im1,im2,h1,h2,h3,h4,h5,h6,h7,h8,h9
            v_abstract,1.ppm,2.ppm,0.7088,-0.010965,-26.07,-0.13602,0.83489,103.19,-0.00023352,-1.5615e-05,1.0004"""
        self.data_type = data_type
        self.hpatch_frame = pd.read_csv(
            os.path.join(root_dir, csv_file), skiprows=skiprows
        )
        self.root_dir = root_dir
        self.transform = transform

        if use_all is not True:
            l = len(self.hpatch_frame)
            s0 = np.int(l * PPT[0]) - 1
            s1 = np.int(l * PPT[1])

            if self.data_type == "train":
                self.hpatch_frame = self.hpatch_frame[:s0]
            elif self.data_type == "eval":
                self.hpatch_frame = self.hpatch_frame[s0:s1]
            elif self.data_type == "test":
                self.hpatch_frame = self.hpatch_frame[s1:]

        print(f"{gct()} : {self.root_dir} has {len(self.hpatch_frame)} pair images")
        self.impair, self.homo, self.imname, self.files = self.generate_data()

    def generate_data(self):
        impair = []
        homo = []
        imname = []
        files = []
        for idx in range(len(self.hpatch_frame)):
            file = self.hpatch_frame.iloc[idx, 0]
            im1_name = self.hpatch_frame.iloc[idx, 1]
            im1_path = os.path.join(self.root_dir, file, im1_name)
            im1 = io.imread(im1_path)

            im2_name = self.hpatch_frame.iloc[idx, 2]
            im2_path = os.path.join(self.root_dir, file, im2_name)
            im2 = io.imread(im2_path)

            files.append(file)
            imname.append([im1_name, im2_name])
            impair.append([im1, im2])

            homo12 = np.asmatrix(
                self.hpatch_frame.iloc[idx, 3:].values.astype("float").reshape(3, 3)
            )
            homo21 = homo12.I
            homo.append([homo12, homo21])
        return impair, homo, imname, files

    def __len__(self):
        assert len(self.impair) == len(self.homo)
        return len(self.impair)

    def __getitem__(self, idx):
        im1 = self.impair[idx][0]
        name1 = self.imname[idx][0]

        im2 = self.impair[idx][1]
        name2 = self.imname[idx][1]

        homo12 = self.homo[idx][0]
        homo21 = self.homo[idx][1]
        sample = {"im1": im1, "im2": im2, "homo12": homo12, "homo21": homo21}

        if self.transform:
            sample = self.transform(sample)

        sample["name1"] = name1
        sample["name2"] = name2
        sample["file"] = self.files[idx]
        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        im1, im2, homo12, homo21 = (
            sample["im1"],
            sample["im2"],
            sample["homo12"],
            sample["homo21"],
        )
        im1_raw, im2_raw = im1, im2

        im1, im1h, im1w, sw1, sh1 = im_rescale(im1, self.output_size)
        im2, im2h, im2w, sw2, sh2 = im_rescale(im2, self.output_size)

        "x and y axes are axis 1 and 0 respectively"
        tform1 = np.mat(
            [[1 / sw1, 0, 0], [0, 1 / sh1, 0], [0, 0, 1]], dtype=homo12.dtype
        )
        tform2 = np.mat([[sw2, 0, 0], [0, sh2, 0], [0, 0, 1]], dtype=homo12.dtype)
        homo12 = tform2 * homo12 * tform1

        tform1 = np.mat([[sw1, 0, 0], [0, sh1, 0], [0, 0, 1]], dtype=homo12.dtype)
        tform2 = np.mat(
            [[1 / sw2, 0, 0], [0, 1 / sh2, 0], [0, 0, 1]], dtype=homo12.dtype
        )
        homo21 = tform1 * homo21 * tform2

        sample["im1"] = im1
        sample["im2"] = im2
        sample["im1_info"] = np.array([sh1, sw1])
        sample["im2_info"] = np.array([sh2, sw2])
        sample["im1_raw"] = im1_raw
        sample["im2_raw"] = im2_raw
        sample["homo12"] = homo12
        sample["homo21"] = homo21

        return sample


class LargerRescale(object):
    """Rescale the image in a sample to a given size which larger than specific size .

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        im1, im2, homo12, homo21 = (
            sample["im1"],
            sample["im2"],
            sample["homo12"],
            sample["homo21"],
        )
        sH, sW = self.output_size
        H1, W1, _ = im1.shape
        H2, W2, _ = im2.shape

        if H1 > sH and H2 > sH and W1 > sW and W2 > sW:
            rescale = Rescale(self.output_size)
            sample = rescale(sample)

        return sample


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        im1, im2, homo12, homo21 = (
            sample["im1"],
            sample["im2"],
            sample["homo12"],
            sample["homo21"],
        )
        new_h, new_w = self.output_size

        h1, w1 = im1.shape[:2]
        h2, w2 = im2.shape[:2]

        # im1 is smaller than expect size
        # don't random crop but rescale
        if h1 <= new_h or w1 <= new_w:
            im1 = transform.resize(im1, (new_h, new_w), mode="constant")
            sw1, sh1 = new_w / w1, new_h / h1
            tform1_12 = np.mat(
                [[1 / sw1, 0, 0], [0, 1 / sh1, 0], [0, 0, 1]], dtype=homo12.dtype
            )
            tform1_21 = np.mat(
                [[sw1, 0, 0], [0, sh1, 0], [0, 0, 1]], dtype=homo12.dtype
            )
        else:
            top1 = np.random.randint(0, h1 - new_h)
            left1 = np.random.randint(0, w1 - new_w)
            im1 = im1[top1 : top1 + int(new_h), left1 : left1 + int(new_w)]
            tform1_12 = np.mat(
                [[1, 0, left1], [0, 1, top1], [0, 0, 1]], dtype=homo12.dtype
            )
            tform1_21 = np.mat(
                [[1, 0, -left1], [0, 1, -top1], [0, 0, 1]], dtype=homo21.dtype
            )

        # same as im2
        if h2 <= new_h or w2 <= new_w:
            im2 = transform.resize(im2, (new_h, new_w), mode="constant")
            sw2, sh2 = new_w / w2, new_h / h2
            tform2_12 = np.mat(
                [[sw2, 0, 0], [0, sh2, 0], [0, 0, 1]], dtype=homo12.dtype
            )
            tform2_21 = np.mat(
                [[1 / sw2, 0, 0], [0, 1 / sh2, 0], [0, 0, 1]], dtype=homo12.dtype
            )
        else:
            top2 = np.random.randint(0, h2 - new_h)
            left2 = np.random.randint(0, w2 - new_w)
            im2 = im2[top2 : top2 + int(new_h), left2 : left2 + int(new_w)]
            tform2_12 = np.mat(
                [[1, 0, -left2], [0, 1, -top2], [0, 0, 1]], dtype=homo12.dtype
            )
            tform2_21 = np.mat(
                [[1, 0, left2], [0, 1, top2], [0, 0, 1]], dtype=homo21.dtype
            )

        homo12 = tform2_12 * homo12 * tform1_12
        homo21 = tform1_21 * homo21 * tform2_21

        sample["im1"] = im1
        sample["im2"] = im2
        sample["homo12"] = homo12
        sample["homo21"] = homo21

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        im1, im1_info, im2, im2_info, homo12, homo21, im1_raw, im2_raw = (
            sample["im1"],
            sample["im1_info"],
            sample["im2"],
            sample["im2_info"],
            sample["homo12"],
            sample["homo21"],
            sample["im1_raw"],
            sample["im2_raw"],
        )

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        im1 = im1.transpose((2, 0, 1))
        im2 = im2.transpose((2, 0, 1))
        im1_raw = im1_raw.transpose((2, 0, 1))
        im2_raw = im2_raw.transpose((2, 0, 1))
        sample["im1"] = torch.from_numpy(im1)
        sample["im2"] = torch.from_numpy(im2)
        sample["im1_info"] = torch.from_numpy(im1_info)
        sample["im2_info"] = torch.from_numpy(im2_info)
        sample["homo12"] = torch.from_numpy(np.asarray(homo12))
        sample["homo21"] = torch.from_numpy(np.asarray(homo21))
        sample["im1_raw"] = torch.from_numpy(im1_raw)
        sample["im2_raw"] = torch.from_numpy(im2_raw)

        return sample


class Normalize(object):
    def __init__(self, mean, std):
        assert isinstance(mean, float)
        assert isinstance(std, float)

        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Parameters: tensor (Tensor) – Numpy image of size (H, W, C) to be normalized.
                    mean (sequence) – Sequence of means for each channel.
                    std (sequence) – Sequence of standard deviations for each channely.
        Returns:    Normalized Tensor image.
        Return type:Tensor
        :param sample: sample dict
        :return: sample dict
        """
        sample["im1"] = (sample["im1"] - self.mean) / self.std
        sample["im2"] = (sample["im2"] - self.mean) / self.std
        return sample


class Grayscale(object):
    def __call__(self, sample):
        """
        Parameters: np.array – np.array image of size (H, W, 1)
        Returns:    gray image
        Return type:np.array
        :param sample: sample list
        :return: sample list
        """
        sample["im1"] = np.expand_dims(color.rgb2gray(sample["im1"]), -1)
        sample["im2"] = np.expand_dims(color.rgb2gray(sample["im2"]), -1)
        return sample
