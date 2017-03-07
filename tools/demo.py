#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test_gallery import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import cPickle

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread(image_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, 'norm', None)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    thresh = 0.05
    inds = np.where(scores[:, 1] > thresh)[0]
    cls_scores = scores[inds, 1]
    cls_boxes = boxes[inds, 4:8]
    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
        .astype(np.float32, copy=False)
    keep = nms(cls_dets, cfg.TEST.NMS)
    cls_dets = cls_dets[keep, :]
    # features = features[keep, :]
    return cls_dets

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    caffe.set_mode_gpu()
    caffe.set_device(0)
    gallery_def = '/home/sy/code/py-faster-rcnn/models/psdb_2/VGG16/test.prototxt'
    caffemodel = '/home/sy/code/py-faster-rcnn/output/psdb_pretrain/VGG16_iter_30000.caffemodel'

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    print '\n\nLoaded network {:s}'.format(caffemodel)
    net = caffe.Net(gallery_def, caffemodel, caffe.TEST)

    path = '/home/sy/code/py-faster-rcnn/demo/'
    # im_names = ['s15535.jpg', 's15536.jpg', 's15537.jpg',
    #             's15538.jpg', 's15539.jpg']
    im_names = ['s15538.jpg']
    for im_name in im_names:
        det = demo(net, path + im_name)
        cache_file = '/home/sy/code/py-faster-rcnn/demo/det.pkl'
        with open(cache_file, 'wb') as fid:
            cPickle.dump(det, fid, cPickle.HIGHEST_PROTOCOL)
        # cache_file = '/home/sy/code/py-faster-rcnn/demo/feat.pkl'
        # with open(cache_file, 'wb') as fid:
        #     cPickle.dump(feat, fid, cPickle.HIGHEST_PROTOCOL)