# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from fast_rcnn.config import cfg
from nms.gpu_nms import gpu_nms
from nms.cpu_nms import cpu_nms, cpu_nms_1d

def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)

def nms_1d(dets, thresh, force_cpu=True):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return cpu_nms_1d(dets, thresh)