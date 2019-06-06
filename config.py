from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
cfg = __C

"""
Project options
"""
__C.PROJ = edict()

# whether us gpu
__C.PROJ.USE_GPU = True

# seed for random
__C.PROJ.SEED = 0

# training, evaluate and test data
__C.PROJ.TRAIN = "view"
__C.PROJ.TRAIN_ALL = False
__C.PROJ.TRAIN_PPT = 0.8

__C.PROJ.EVAL = "view"
__C.PROJ.EVAL_ALL = False
__C.PROJ.EVAL_PPT = 0.1

__C.PROJ.TEST = "view"
__C.PROJ.TEST_ALL = False
__C.PROJ.TEST_PPT = 0.1

"""
Model options
"""
__C.MODEL = edict()

# gaussian kernel size
__C.MODEL.GAUSSIAN_KSIZE = 15

# gaussian kernel sigma
__C.MODEL.GAUSSIAN_SIGMA = 0.5

# Descriptor Threshold
__C.MODEL.DES_THRSH = 1.0

# Coordinate Threshold
__C.MODEL.COO_THRSH = 5.0

# Ksize
__C.MODEL.KSIZE = 3

# padding
__C.MODEL.padding = 1

# dilation
__C.MODEL.dilation = 1

# scale_list
__C.MODEL.scale_list = [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]

"""
Loss options
"""
__C.LOSS = edict()

# score loss wight
__C.LOSS.SCORE = 1000

# pair loss weight
__C.LOSS.PAIR = 1

"""
Training options
"""
__C.TRAIN = edict()

# batch size
__C.TRAIN.BATCH_SIZE = 1

# Train epoch
__C.TRAIN.EPOCH_NUM = 201

# Train log interval
__C.TRAIN.LOG_INTERVAL = 5

# weight decay
__C.TRAIN.WEIGHT_DECAY = 1e-4

# detector learning rate
__C.TRAIN.DET_LR = 0.1

# descriptor learning rate
__C.TRAIN.DES_LR = 10

# detection optimizer (adam/sgd)
__C.TRAIN.DET_OPTIMIZER = "adam"

# adjust detection lr (sgd/exp)
__C.TRAIN.DET_LR_SCHEDULE = "exp"

# detector weight decay
__C.TRAIN.DET_WD = 0

# descriptor optimizer (adam/sgd)
__C.TRAIN.DES_OPTIMIZER = "adam"

# adjust descriptor lr (sgd/exp)
# __C.TRAIN.DES_LR_SCHEDULE = 'exp'
__C.TRAIN.DES_LR_SCHEDULE = "sgd"

# descriptor weight decay
__C.TRAIN.DES_WD = 0

# learning rate decay epoch
__C.TRAIN.LR_DECAY_EPOCH = 5

# learning rate base line
__C.TRAIN.LR_BASE = 0.0001

# score strength weight
__C.TRAIN.score_com_strength = 100.0

# scale strength weight
__C.TRAIN.scale_com_strength = 100.0

# non maximum supression threshold
__C.TRAIN.NMS_THRESH = 0.0

# nms kernel size
__C.TRAIN.NMS_KSIZE = 5

# top k patch
__C.TRAIN.TOPK = 512

"""
Image data options
"""
# View train sequence Mean and Std
__C.view = edict()
__C.view.csv = "hpatch_view.csv"
__C.view.root = "../data/hpatch_v_sequence"
__C.view.MEAN = 0.4230204841414801
__C.view.STD = 0.25000138349993173
__C.view.NUM = 295

# illumination sequence Mean and Std
__C.illu = edict()
__C.illu.csv = "hpatch_illum.csv"
__C.illu.root = "../data/hpatch_i_sequence"
__C.illu.MEAN = 0.4337542740124942
__C.illu.STD = 0.2642307153894012
__C.illu.NUM = 285

# illumination sequence Mean and Std
__C.ef = edict()
__C.ef.csv = "EFDataset.csv"
__C.ef.root = "../data/EFDataset"
__C.ef.MEAN = 0.4630827743610772
__C.ef.STD = 0.24659232013004403
__C.ef.NUM = 293

"""
Patch options
"""
__C.PATCH = edict()

# patch size
__C.PATCH.SIZE = 32


"""
Hardnet options
"""
__C.HARDNET = edict()

# margin for hardnet loss
__C.HARDNET.MARGIN = 1.0
