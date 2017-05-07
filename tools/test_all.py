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

import torch
import cv2
import network
from phone_net import PhoneNet
from utils.timer import Timer
from fast_rcnn.nms_wrapper import nms
from progressbar import ProgressBar
import sys

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


def im_detect(net, image):
    im_data = net.get_image_blob(image)
    # print im_data.shape
    cls_prob = net(im_data)
    scores = [cls_prob[i].data.cpu().numpy() for i in range(13)]
    return scores


def safe_log(x, minval=10e-40):
    return np.log(x.clip(min=minval))

if __name__ == '__main__':

    all_boxes = None

    test_def = os.path.join(os.getcwd(), 'models', 'express', 'VGG16', 'test.prototxt')
    caffemodel = os.path.join(os.getcwd(), 'output', 'express_train', 'express_iter_25000.caffemodel')
    imdb_name = 'express_test'
    cfg_file = 'experiments/cfgs/train.yml'
    cfg.GPU_ID = 0
    cfg_from_file(cfg_file)
    print 'Using config:'
    pprint.pprint(cfg)

    imdb = get_imdb(imdb_name, os.path.join(cfg.DATA_DIR, 'express'), ratio=0)
    output_dir = os.path.join(os.getcwd(), 'output', 'test_all')
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
        all_boxes = test_net(net, imdb, output_dir, thresh=0.8, iou_thres=0.5)
    print '###### getting detections'
    det_phone_according_to_thres = True
    # imdb.get_detections(all_boxes, output_dir, iou_thres=0.5)
    imdb.get_detections_thres(all_boxes, output_dir, thres=0.8, iou_thres=0.5)

    if not det_phone_according_to_thres:
        print '###### starting phone testing'
        imdb_name = 'phone_test'
        cfg_file = 'experiments/cfgs/train_phone.yml'
        model_path = 'output/phone_train/'
        model_name = 'phone_30000.h5'
        trained_model = model_path + model_name

        rand_seed = 1024
        max_per_image = 300
        thresh = 0.05
        vis = False

        if rand_seed is not None:
            np.random.seed(rand_seed)
        cfg_from_file(cfg_file)
        print 'Using config:'
        pprint.pprint(cfg)

        imdb = get_imdb(imdb_name, output_dir, ratio=0)
        
        cache_file = os.path.join(output_dir, 'detection_score.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                all_boxes = cPickle.load(fid)
            print "###### load detection_score"
        else:
            # load data
            print '###### loading model {}'.format(trained_model)
            net = PhoneNet(classes=imdb.classes, debug=False)
            network.load_net(trained_model, net)
            print '###### load model successfully!'

            net.cuda()
            net.eval()

            print '###### starting test ...'
            num_images = len(imdb.image_index)
            all_boxes = [[[] for _ in xrange(13)]
                         for _ in xrange(num_images)]
            pbar = ProgressBar(maxval=num_images)
            pbar.start()

            for i in range(num_images):
                im = cv2.imread(imdb.image_path_at(i))
                scores = im_detect(net, im)
                for j in range(13):
                    all_boxes[i][j] = scores[j]
                pbar.update(i)
            pbar.finish()

            with open(cache_file, 'wb') as fid:
                cPickle.dump(all_boxes, fid, cPickle.HIGHEST_PROTOCOL)

        print '###### Evaluating detections'
        imdb.evaluate_detections(all_boxes, output_dir, weights=np.ones((12, 10)))
    else:
        net = PhoneNet()
        trained_model = os.path.join(os.getcwd(), 'output', 'phone_out', '11', 'phone_25000.h5')
        network.load_net(trained_model, net)
        net.cuda()
        net.eval()
        print '\nLoaded network {:s}'.format(trained_model)

        phone_path = os.path.join(os.getcwd(), 'output', 'test_all', 'images')
        if not os.path.isdir(phone_path):
            print 'error data dir path {}'.format(phone_path)
        filelist = sorted(os.listdir(phone_path))
        if not filelist:
            print 'no pic in dir {}'.format(phone_path)

        phone_all = {}
        score_all = {}
        cache_file = os.path.join(os.getcwd(), 'output', 'test_all', 'phone_all.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                phone_all, score_all = cPickle.load(fid)
            print "load phone_all"
        else:
            for im_name in filelist:
                im = cv2.imread(phone_path + '/' + im_name)
                scores = im_detect(net, im)

                phone_length = np.argmax(scores[-1]) + 5
                scores = np.vstack((scores[:phone_length]))
                scores = safe_log(scores[:, :-1])
                res = np.argmax(scores, axis=1)
                score = np.max(scores, axis=1)
                if np.sum(score) / score.shape[0] < -0.12:
                    continue
                phone_all[im_name] = res
                score_all[im_name] = score
            with open(cache_file, 'wb') as fid:
                cPickle.dump([phone_all, score_all], fid, cPickle.HIGHEST_PROTOCOL)
            print 'save phone_all'
        print 'detected phone nums: {}'.format(len(phone_all.keys()))

        namelist, info = None, None
        namelist_path = os.path.join(os.getcwd(), 'output', 'test_all', 'namelist.pkl')
        if os.path.exists(namelist_path):
            with open(namelist_path, 'rb') as fid:
                namelist = cPickle.load(fid)
        info_path = os.path.join(os.getcwd(), 'output', 'test_all', 'info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as fid:
                info = cPickle.load(fid)

        phone_num, phone_tp = 0, 0
        for im_name in namelist:
            gt_phone, filename = info[im_name][0], info[im_name][1]
            det_phone = [phone_all[name] for name in filename if name in phone_all]
            for gt in gt_phone:
                for det in det_phone:
                    if (gt.shape[0] == det.shape[0]) and (gt == det).all():
                        phone_tp += 1
                        break
                phone_num += 1
        print 'phone_acc: {} / {} = {}'.format(phone_tp, phone_num, float(phone_tp)/phone_num)