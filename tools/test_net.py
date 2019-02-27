import _init_paths
from fast_rcnn.test import test_net
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import cPickle
import caffe
import argparse
import pprint
import time, os, sys
import numpy as np

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

if __name__ == '__main__':
    all_boxes = None

    test_def = os.path.join(os.getcwd(), 'models', 'express', 'VGG16_4_multi', 'test.prototxt')
    caffemodel = os.path.join(os.getcwd(), 'output', 'express_train', 'VGG16_4_multi', 'express_iter_15000.caffemodel')
    imdb_name = 'express_test'
    cfg_file = 'experiments/cfgs/train.yml'
    cfg.GPU_ID = 0
    cfg_from_file(cfg_file)
    print 'Using config:'
    pprint.pprint(cfg)

    imdb = get_imdb(imdb_name, os.path.join(cfg.DATA_DIR, 'express'), ratio=0.8)
    output_dir = os.path.join(os.getcwd(), 'output', 'express_test')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    cache_file = os.path.join(output_dir, 'detections.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
        print "###### load detections"
    else:
        caffe.set_mode_gpu()
        caffe.set_device(0)
        net = caffe.Net(test_def, caffemodel, caffe.TEST)
        net.name = os.path.splitext(os.path.basename(caffemodel))[0]
        all_boxes = test_net(net, imdb, output_dir, thresh=0.1, iou_thres=0.6)
    imdb.evaluate_detections(all_boxes, output_dir, iou_thres=0.5)

    ap = 0
    for item in [x / 100. for x in range(50,100,5)]:
        ap += imdb.evaluate_detections(all_boxes, output_dir, iou_thres=item)
    print 'map[0.5,0.95]: ', ap/10

