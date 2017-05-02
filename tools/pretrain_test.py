import _init_paths
import os
import torch
import cv2
import cPickle
import numpy as np
from collections import Counter

import network
from pretrain_phone import Net
from utils.timer import Timer
from fast_rcnn.nms_wrapper import nms
import pprint


from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from progressbar import ProgressBar


def prepare_roidb(imdb):
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)

def im_detect(net, image):

    im_data = net.get_image_blob(image)
    # print im_data.shape
    cls_prob = net(im_data)
    scores = cls_prob.data.cpu().numpy()
    return scores

def phone_append(a):
    if a.shape[0] < 12:
        res = np.zeros((12, ))
        res[:] = 10
        res[:a.shape[0]] = a
        return res.astype(np.uint8)
    return a[:].astype(np.uint8)

if __name__ == '__main__':
    # hyper-parameters
    imdb_name = 'mnist_test'
    cfg_file = 'experiments/cfgs/train_mnist.yml'
    model_path = 'output/mnist_train/'
    model_name = 'mnist_4200.h5'
    trained_model = model_path + model_name

    rand_seed = 1024
    max_per_image = 300
    thresh = 0.05
    vis = False

    if rand_seed is not None:
        np.random.seed(rand_seed)
    cfg_from_file(cfg_file)
    # print 'Using config:'
    # pprint.pprint(cfg)

    # imdb = get_imdb(imdb_name)
    imdb = get_imdb(imdb_name, os.path.join(cfg.DATA_DIR, 'express', 'pretrain_mnist'), ratio=0.99)
    prepare_roidb(imdb)
    
    output_dir = get_output_dir(imdb, model_name)
    cache_file = os.path.join(output_dir, 'detection_score.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            all_labels = cPickle.load(fid)
        print "load detection_score"
    else:
        # load data
        print 'loading model {}'.format(trained_model)
        net = Net()
        network.load_net(trained_model, net)
        print 'load model successfully!'

        net.cuda()
        net.eval()

        print 'starting test ...'
        num_images = len(imdb.image_index)
        all_labels = []
        pbar = ProgressBar(maxval=num_images)
        pbar.start()

        for i in range(num_images):
            im = cv2.imread(imdb.image_path_at(i))
            scores = im_detect(net, im)
            all_labels.append(scores)
            pbar.update(i)
        pbar.finish()

        with open(cache_file, 'wb') as fid:
            cPickle.dump(all_labels, fid, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating detections'
    # imdb.evaluate_detections(all_labels, output_dir)
    imdb.evaluate_detections(all_labels, output_dir)


    