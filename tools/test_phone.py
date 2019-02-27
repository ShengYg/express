import _init_paths
import cv2
import os
import torch
import cPickle
import numpy as np
from collections import Counter

import network
from phone_net import PhoneNet
from utils.timer import Timer
from fast_rcnn.nms_wrapper import nms
import pprint

from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from progressbar import ProgressBar


def prepare_roidb(imdb):
    roidb = imdb.gt_roidb()
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
    return roidb

def im_detect(net, image, height=48, width=256):
    im_data = net.get_image_blob(image, height=height, width=width)
    cls_prob = net(im_data)
    scores = [cls_prob[i].data.cpu().numpy() for i in range(13)]
    return scores

def phone_append(a):
    if a[0].shape[0] < 12:
        res = np.zeros((12, ))
        res[:] = 10
        res[:a[0].shape[0]] = a[0]
        return res.astype(np.uint8)
    return a[0][:].astype(np.uint8)

if __name__ == '__main__':
    # hyper-parameters
    imdb_name = 'phone_test'
    model_path = 'output/phone_train/TIME-20190223-092831/'
    model_name = 'phone_20000.h5'
    trained_model = model_path + model_name

    rand_seed = 1024
    save_name = model_name[:-3]
    max_per_image = 300
    thresh = 0.05
    vis = False


    # imdb = get_imdb(imdb_name, os.path.join(cfg.DATA_DIR, 'express', 'pretrain_db_benchmark'), ratio=0.8)
    # imdb = get_imdb(imdb_name, os.path.join(cfg.DATA_DIR, 'express', 'pretrain_db_benchmark_extra'), ratio=0.8)
    imdb = get_imdb(imdb_name, os.path.join(cfg.DATA_DIR, 'express', 'test_db_benchmark'), ratio=0.)
    roidb = prepare_roidb(imdb)

    # loading rescaling weights
    # info = None
    # cache_file = '/home/sy/code/re_id/express/data/express/pretrain_db_benchmark/info.pkl'
    # if os.path.exists(cache_file):
    #     with open(cache_file, 'rb') as fid:
    #         info = cPickle.load(fid)
    # phones = info.values()
    # length_weights = [item[0].shape[0] for item in phones]
    # length_weights = np.array([Counter(length_weights)[i] for i in range(5, 13)])
    # phones = np.vstack(map(phone_append, phones))
    # phones = [phones[:, i] for i in range(12)]
    # weights = np.vstack([np.array([Counter(phones[i])[j] for j in range(10)])[:10] for i in range(12)])
    length_weights = np.ones((8,))
    weights = np.ones((12, 10))
    
    output_dir = get_output_dir(imdb, model_name)
    cache_file = os.path.join(output_dir, 'detection_score.pkl')
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(13)]
                 for _ in xrange(num_images)]
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
        print "load detection_score"
    else:
        # load data
        print 'loading model {}'.format(trained_model)
        net = PhoneNet(classes=imdb.classes, bn=False)
        network.load_net(trained_model, net)
        print 'load model successfully!'

        net.cuda()
        net.eval()
        print 'starting test ...'
        pbar = ProgressBar(maxval=num_images)
        pbar.start()
        for i in range(num_images):
            im = cv2.imread(imdb.image_path_at(i))
            x, y, w, h = roidb[i]['bbox']
            im = im[y:y+h, x:x+w, :]
            scores = im_detect(net, im, height=48, width=256)
            for j in range(13):
                all_boxes[i][j] = scores[j]
            pbar.update(i)
        pbar.finish()

        with open(cache_file, 'wb') as fid:
            cPickle.dump(all_boxes, fid, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    # imdb.evaluate_ohem(all_boxes, output_dir, roidb)
    imdb.evaluate_detections(all_boxes, output_dir, roidb, weights=weights, length_weights=length_weights)


    