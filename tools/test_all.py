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

def voteclassifier(res_list, score_list):
    # res_list: [[1,2,3], [...], ...]
    d = {}
    for res, score in zip(res_list, score_list):
        res = ''.join([str(i) for i in res])
        if res in d:
            d[res].append(score)
        else:
            d[res] = [score]
    res, maxinum, score = '', 0, None
    for k,v in d.items():
        if len(v) > maxinum:
            maxinum = len(v)
            res = k
            score = np.mean(v)
    return np.array([int(i) for i in list(res)]), score



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

def im_detect_list(net, image_list):
    im_data = net.get_image_blob_list(image_list)
    cls_prob = net(im_data)
    scores = [cls_prob[i].data.cpu().numpy() for i in range(13)]
    return scores


def safe_log(x, minval=10e-40):
    return np.log(x.clip(min=minval))

if __name__ == '__main__':

    all_boxes = None
    test_def = os.path.join(os.getcwd(), 'models', 'express', 'VGG16_4', 'test.prototxt')
    caffemodel = os.path.join(os.getcwd(), 'output', 'express_train', 'VGG16_4', 'express_iter_15000.caffemodel')
    # test_def = os.path.join(os.getcwd(), 'models', 'express', 'VGG16_4_multi_RPN', 'test.prototxt')
    # caffemodel = os.path.join(os.getcwd(), 'output', 'express_train', 'VGG16_4_multi_RPN', 'express_iter_25000.caffemodel')
    imdb_name = 'express_test'
    cfg_file = 'experiments/cfgs/train.yml'
    cfg.GPU_ID = 0
    cfg_from_file(cfg_file)
    # print 'Using config:'
    # pprint.pprint(cfg)

    imdb = get_imdb(imdb_name, os.path.join(cfg.DATA_DIR, 'express'), ratio=0)
    output_dir = os.path.join(os.getcwd(), 'output', 'test_all')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
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
        all_boxes = test_net(net, imdb, output_dir, thresh=0.1, iou_thres=0.5)

    version1 = False    # version1 is the same with version2, except that it will crop image in test_all for debug
    RANDOM_EXPAND = False
    if version1:
        print '###### getting detections'
        imdb.get_detections_thres(all_boxes, output_dir, thres=0.1)

        print '###### starting recognition'
        net = PhoneNet()
        # trained_model = os.path.join(os.getcwd(), 'output', 'phone_train', \
        #                 'TIME-20190223-162248', 'phone_16000.h5')       # without random expanding
        trained_model = os.path.join(os.getcwd(), 'output', 'phone_train', \
                        'TIME-20190223-092831', 'phone_20000.h5')       # with random expanding
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
            for cropped_name in filelist:
                im = cv2.imread(phone_path + '/' + cropped_name)
                # print im.shape
                scores = im_detect(net, im)

                phone_length = np.argmax(scores[-1]) + 5
                scores = np.vstack((scores[:phone_length]))
                scores = safe_log(scores[:, :-1])
                res = np.argmax(scores, axis=1)
                score = np.max(scores, axis=1)
                if np.sum(score) / score.shape[0] < -0.10:
                    continue
                phone_all[cropped_name] = res
                score_all[cropped_name] = score
            with open(cache_file, 'wb') as fid:
                cPickle.dump([phone_all, score_all], fid, cPickle.HIGHEST_PROTOCOL)
            print 'save phone_all'
        print 'recognized phone nums: {}'.format(len(phone_all.keys()))

        namelist, info = None, None
        namelist_path = os.path.join(os.getcwd(), 'output', 'test_all', 'namelist.pkl')
        if os.path.exists(namelist_path):
            with open(namelist_path, 'rb') as fid:
                namelist = cPickle.load(fid)
        info_path = os.path.join(os.getcwd(), 'output', 'test_all', 'info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as fid:
                info = cPickle.load(fid)

        # calculate tp, precision, recall in each image, and average them
        pred_all, recall_all = 0, 0
        for express_name in namelist:
            gt_phone, cropped_name = info[express_name][0], info[express_name][1]
            det_phone = [phone_all[name] for name in cropped_name if name in phone_all]
            tp = 0
            for gt in gt_phone:
                for det in det_phone:
                    if (gt.shape[0] == det.shape[0]) and (gt == det).all():
                        tp += 1
                        break
            if len(det_phone) != 0:
                pred_all += float(tp) / len(det_phone)
            recall_all += float(tp) / len(gt_phone)

                # phone_num += 1
        # print 'express num: {}'.format(len(namelist))
        print 'phone precision: {}'.format(pred_all/len(namelist))
        print 'phone recall: {}'.format(recall_all/len(namelist))

        # calculate tp in all image, and calculate precision, recall
        det_all, gt_all, tp_all = 0, 0, 0
        all_name = []
        for express_name in namelist:
            gt_phone, cropped_name = info[express_name][0], info[express_name][1]
            det_phone = [phone_all[name] for name in cropped_name if name in phone_all]
            all_name.extend([name for name in cropped_name if name in phone_all])
            for gt in gt_phone:
                for det in det_phone:
                    if (gt.shape[0] == det.shape[0]) and (gt == det).all():
                        tp_all += 1
                        break
            det_all += len(det_phone)
            gt_all += len(gt_phone)
            
        # print 'express num: {}'.format(len(namelist))
        print 'phone precision: {} / {} = {}'.format(tp_all, det_all, float(tp_all)/det_all)
        print 'phone recall: {} / {} = {}'.format(tp_all, gt_all, float(tp_all)/gt_all)

        cache_file = os.path.join('temp', 'temp.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(all_name, fid, cPickle.HIGHEST_PROTOCOL)
    else:
        print '###### starting recognition'
        net = PhoneNet()
        # trained_model = os.path.join(os.getcwd(), 'output', 'phone_train', \
        #                 'TIME-20190223-162248', 'phone_16000.h5')       # without random expanding
        trained_model = os.path.join(os.getcwd(), 'output', 'phone_train', \
                        'TIME-20190223-092831', 'phone_20000.h5')       # with random expanding
        network.load_net(trained_model, net)
        net.cuda()
        net.eval()
        print '\nLoaded network {:s}'.format(trained_model)


        gt_roidb = imdb.gt_roidb()
        print 'gt image nums: {}'.format(len(gt_roidb))
        assert len(all_boxes) == 2
        assert len(all_boxes[1]) == len(gt_roidb)

        pbar = ProgressBar(maxval=len(gt_roidb))
        pbar.start()
        k = 0
        pred_each, recall_each = 0, 0
        det_all, gt_all, tp_all = 0, 0, 0
        for gt, det in zip(gt_roidb, all_boxes[1]):
            det = np.asarray(det)
            im_width = gt['im_size'][0]
            im_name = gt['image']
            gt_phone = gt['label']

            inds = np.where(det[:, 0] < im_width*0.4)[0]
            det = det[inds]
            inds = np.where(det[:, -1] > 0.8)[0]    # det threshold
            det = det[inds]
            keep = nms(det, 0.)
            det = det[keep, :]
            det = det[:3]

            im = cv2.imread(os.path.join(imdb._data_path, im_name))
            width = im.shape[1]
            det_phone = []
            for box in det:
                x1, y1, x2, y2 = box[:4].astype(np.int16)

                if RANDOM_EXPAND:
                    x1 = max(x1 - int(5 * width / 300.0), 0)
                    x2 = max(x2 + int(5 * width / 300.0), 0)
                    y1 = max(y1 - 3, 0)
                    y2 = max(y2 + 3, 0)

                    cropped_list = []
                    for shift_grid in range(-10, 11):
                        shift = int(shift_grid * width / 600.0)
                        cropped = im[y1:y2, max(x1+shift, 0):max(x2+shift, 0), :]
                        cropped_list.append(cropped)
      
                    scores = im_detect_list(net, cropped_list)

                    res_list = []
                    sco_list = []
                    for i in range(scores[0].shape[0]):
                        phone_length = np.argmax(scores[-1][i]) + 5
                        score = np.vstack([scores[j][i] for j in range(phone_length)])
                        score = safe_log(score[:, :-1]).astype(np.float32)
                        res = np.argmax(score, axis=1)
                        sco = np.max(score, axis=1)
                        res_list.append(res)
                        sco_list.append(np.mean(sco))
                    res_after_select, score_after_select = voteclassifier(res_list, sco_list)
                    if score_after_select < -0.1:       ## score threshold
                        continue
                    det_phone.append(res_after_select)
                else:
                    cropped = im[y1:y2, x1:x2, :]
                    scores = im_detect(net, cropped)

                    phone_length = np.argmax(scores[-1]) + 5
                    scores = np.vstack((scores[:phone_length]))
                    scores = safe_log(scores[:, :-1])
                    res = np.argmax(scores, axis=1)
                    score = np.max(scores, axis=1)
                    if np.sum(score) / score.shape[0] < -0.1:    #score threshold
                        continue
                    det_phone.append(res)

            tp_each = 0
            for gt in gt_phone:
                for det in det_phone:
                    if (gt.shape[0] == det.shape[0]) and (gt == det).all():
                        tp_each += 1
                        tp_all += 1
                        break
            if len(det_phone) != 0:
                pred_each += float(tp_each) / len(det_phone)
            recall_each += float(tp_each) / len(gt_phone)
            det_all += len(det_phone)
            gt_all += len(gt_phone)
            k += 1
            pbar.update(k)
        pbar.finish()
        print 'phone precision: {}'.format(pred_each/len(gt_roidb))
        print 'phone recall: {}'.format(recall_each/len(gt_roidb))
        print 'phone precision: {} / {} = {}'.format(tp_all, det_all, float(tp_all)/det_all)
        print 'phone recall: {} / {} = {}'.format(tp_all, gt_all, float(tp_all)/gt_all)

        ################################################################################################################

        RANDOM_EXPAND = True

        print '###### starting recognition'
        net = PhoneNet()
        # trained_model = os.path.join(os.getcwd(), 'output', 'phone_train', \
        #                 'TIME-20190223-162248', 'phone_16000.h5')       # without random expanding
        trained_model = os.path.join(os.getcwd(), 'output', 'phone_train', \
                        'TIME-20190223-092831', 'phone_20000.h5')       # with random expanding
        network.load_net(trained_model, net)
        net.cuda()
        net.eval()
        print '\nLoaded network {:s}'.format(trained_model)


        gt_roidb = imdb.gt_roidb()
        print 'gt image nums: {}'.format(len(gt_roidb))
        assert len(all_boxes) == 2
        assert len(all_boxes[1]) == len(gt_roidb)

        pbar = ProgressBar(maxval=len(gt_roidb))
        pbar.start()
        k = 0
        pred_each, recall_each = 0, 0
        det_all, gt_all, tp_all = 0, 0, 0
        for gt, det in zip(gt_roidb, all_boxes[1]):
            det = np.asarray(det)
            im_width = gt['im_size'][0]
            im_name = gt['image']
            gt_phone = gt['label']

            inds = np.where(det[:, 0] < im_width*0.4)[0]
            det = det[inds]
            inds = np.where(det[:, -1] > 0.8)[0]    # det threshold
            det = det[inds]
            keep = nms(det, 0.)
            det = det[keep, :]
            det = det[:3]

            im = cv2.imread(os.path.join(imdb._data_path, im_name))
            width = im.shape[1]
            det_phone = []
            for box in det:
                x1, y1, x2, y2 = box[:4].astype(np.int16)

                if RANDOM_EXPAND:
                    x1 = max(x1 - int(5 * width / 300.0), 0)
                    x2 = max(x2 + int(5 * width / 300.0), 0)
                    y1 = max(y1 - 3, 0)
                    y2 = max(y2 + 3, 0)

                    cropped_list = []
                    for shift_grid in range(-10, 11):
                        shift = int(shift_grid * width / 600.0)
                        cropped = im[y1:y2, max(x1+shift, 0):max(x2+shift, 0), :]
                        cropped_list.append(cropped)
      
                    scores = im_detect_list(net, cropped_list)

                    res_list = []
                    sco_list = []
                    for i in range(scores[0].shape[0]):
                        phone_length = np.argmax(scores[-1][i]) + 5
                        score = np.vstack([scores[j][i] for j in range(phone_length)])
                        score = safe_log(score[:, :-1]).astype(np.float32)
                        res = np.argmax(score, axis=1)
                        sco = np.max(score, axis=1)
                        res_list.append(res)
                        sco_list.append(np.mean(sco))
                    res_after_select, score_after_select = voteclassifier(res_list, sco_list)
                    if score_after_select < -0.1:       ## score threshold
                        continue
                    det_phone.append(res_after_select)
                else:
                    cropped = im[y1:y2, x1:x2, :]
                    scores = im_detect(net, cropped)

                    phone_length = np.argmax(scores[-1]) + 5
                    scores = np.vstack((scores[:phone_length]))
                    scores = safe_log(scores[:, :-1])
                    res = np.argmax(scores, axis=1)
                    score = np.max(scores, axis=1)
                    if np.sum(score) / score.shape[0] < -0.1:    #score threshold
                        continue
                    det_phone.append(res)

            tp_each = 0
            for gt in gt_phone:
                for det in det_phone:
                    if (gt.shape[0] == det.shape[0]) and (gt == det).all():
                        tp_each += 1
                        tp_all += 1
                        break
            if len(det_phone) != 0:
                pred_each += float(tp_each) / len(det_phone)
            recall_each += float(tp_each) / len(gt_phone)
            det_all += len(det_phone)
            gt_all += len(gt_phone)
            k += 1
            pbar.update(k)
        pbar.finish()
        print 'phone precision: {}'.format(pred_each/len(gt_roidb))
        print 'phone recall: {}'.format(recall_each/len(gt_roidb))
        print 'phone precision: {} / {} = {}'.format(tp_all, det_all, float(tp_all)/det_all)
        print 'phone recall: {} / {} = {}'.format(tp_all, gt_all, float(tp_all)/gt_all)
