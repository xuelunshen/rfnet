# RF-Net: An End-to-End Image Matching Network based on Receptive Field

This repository is a Pytorch implementation for

> Xuelun Shen, Cheng Wang, Xin Li, Zenglei Yu, Jonathan Li, Chenglu Wen, Ming Cheng, Zijian He. "RF-Net: An End-to-End Image Matching Network based on Receptive Field." In the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 8132-8140.

If you use this code in your research, please cite [the paper](https://arxiv.org/abs/1906.00604).

---

## Environment

This code is based on `Python (3.6.2)` and `Pytorch (py36_cuda9.2.148_cudnn7.1.4_1)`, and tested on `Ubuntu 16.04` with `GeForce RTX 2080 Ti`.

Necessary packages are listed in `requirements.txt`, and please install it by

```bash
pip install -r requirements.txt
```

---

## Usage

### Get the data

Download datasets from [this link](https://drive.google.com/open?id=1CyoJiY8d-byyZxOzkwhQ-MG-viRQGkOd) and unzip it, then you will get a `data` folder.

### Get pretrained model

Download pretrained model from [this link](https://drive.google.com/open?id=1kGFH3gjrEBAsoWNO7zcOVCR6JxBluMDN) and unzip it, then you will get a `runs` folder.

### Prepare directory structure

Put the `data` and `runs` folders to the same directory same as `rfnet` folder.

The directory structure should be like this:

```bash
project
│
└───rfnet
│   │   config.py
│   │   hpatch_dataset.py
│   │   ms.py
│   │   ...
│   │
│   └───model
│   │   │   rf_des.py
│   │   │   rf_det_module.py
│   │   │   rf_det_so.py
│   │   │   ...
│   │   │
│   └───utils
│       │   common_utils.py
│       │   eval_utils.py
│       │   hpatch_to_csv.py
│       │   ...
│
└───data
|   │
|   └───hpatch_v_sequence
|   │
|   └───hpatch_i_sequence
|   │
|   └───EFDataset
|
└───runs

```

## Training

```bash
# the number 0 means the first GPU on your machine.
sh train.sh 0
```

## Evaluation

```bash
# the number 0 means the first GPU on your machine.
# parameter 'root' represents the path to your runs folder
# parameter 'date' represents the date when you saved your model
# parameter 'model' represents the model name
sh ms.sh 0 $root $date $model

# if you try to run the pretrained model
# the command should be like:
sh ms.sh 0 /home/sxl/DL/runs 10_24_09_25 e121_NN_0.480_NNT_0.655_NNDR_0.813_MeanMS_0.649.pth.tar
```

## Acknowledgement

We would like to thank

> EF dataset [1]

> HPatches dataset [2]

for providing the image data.

> [1] Zitnick, C.L., & Ramnath, K. (2011). Edge foci interest points. 2011 International Conference on Computer Vision (ICCV), 359-366.

> [2] Balntas, V., Lenc, K., Vedaldi, A., & Mikolajczyk, K. (2017). HPatches: A Benchmark and Evaluation of Handcrafted and Learned Local Descriptors. 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 3852-3861.
