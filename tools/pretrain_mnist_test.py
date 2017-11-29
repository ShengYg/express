import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
import numpy as np
import sys
import cPickle
import os
import cv2
from progressbar import ProgressBar

from utils.timer import Timer

def prepare_roidb(imdb):
    roidb = imdb.gt_roidb()
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
    return roidb

def _get_image_blob(im):
    im = im.astype(np.float32, copy=False)
    im -= cfg.PIXEL_MEANS
    im_shape = im.shape
    im_scale_x = float(cfg.TRAIN.WIDTH) / float(im_shape[1])
    im_scale_y = float(cfg.TRAIN.HEIGHT) / float(im_shape[0])

    im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y,
                    interpolation=cv2.INTER_LINEAR)

    blob = np.zeros((1, cfg.TEST.HEIGHT, cfg.TEST.WIDTH, 3), dtype=np.float32)
    blob[0, :, :, :] = im
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def _get_blobs(im):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None,}
    blobs['data'] = _get_image_blob(im)
    return blobs

def im_detect(net, im):
    
    blobs = _get_blobs(im)

    net.blobs['data'].reshape(*(blobs['data'].shape))
    forward_kwargs = {'data': blobs['data'].astype(np.float32, copy=False)}
    blobs_out = net.forward(**forward_kwargs)
    scores = blobs_out['cls_prob']

    return scores

def test_net(net, imdb, output_dir, roidb):
    num_images = len(imdb.image_index)
    all_labels = []

    pbar = ProgressBar(maxval=num_images)
    pbar.start()
    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        scores = im_detect(net, im)
        all_labels.append(np.argmax(scores[0]))
        pbar.update(i)
    pbar.finish()

    cache_file = os.path.join(output_dir, 'detections.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(all_labels, fid, cPickle.HIGHEST_PROTOCOL)

    imdb.evaluate_detections(all_labels, output_dir, roidb)

if __name__ == '__main__':
    caffemodel = '/home/sy/code/re_id/express/output/mnist_train/mnist_iter_12000.caffemodel'
    test_def = '/home/sy/code/re_id/express/models/mnist/VGG16/test.prototxt'
    imdb_name = 'mnist_test'
    cfg_file = '/home/sy/code/re_id/express/experiments/cfgs/train_mnist.yml'
    cfg_from_file(cfg_file)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(0)

    imdb = get_imdb(imdb_name, os.path.join('/home/sy/code/re_id/express/data', 'express', 'pretrain_mnist'), ratio=0.95)
    print 'Loaded dataset {} for training'.format(imdb.name)
    roidb = prepare_roidb(imdb)
    print 'roidb length: {}'.format(len(imdb.roidb))

    cache_file = os.path.join('/home/sy/code/re_id/express/data/cache', 'mnist_test_gt_roidb.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)

    output_dir = get_output_dir(imdb, os.path.splitext(os.path.basename(caffemodel))[0])
    print output_dir
    net = caffe.Net(test_def, caffemodel, caffe.TEST)
    test_net(net, imdb, output_dir, roidb)