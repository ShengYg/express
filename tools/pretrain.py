import _init_paths
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from datetime import datetime
import pprint
import cv2

import network
from utils.timer import Timer
import time

from roi_data_layer.mnist_layer import MnistDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
from tensorboard_logger import configure, log_value
import logging

from utils.blob import im_list_to_blob
from fast_rcnn.config import cfg
import network
from network import Conv2d, FC

from termcolor import cprint

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

def prepare_roidb(imdb):
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)

class Net(nn.Module):
    def __init__(self, bn=False):
        super(Net, self).__init__()
        cnn = nn.Sequential()

        def convRelu(i, nIn, nOut, k, s, p, bn=False):
            cnn.add_module('conv{0}'.format(i), nn.Conv2d(nIn, nOut, k, s, p))
            if bn:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, 1, 64, 3, 1, 1)
        convRelu(1, 64, 64, 3, 1, 1)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 24*12
        convRelu(2, 64, 128, 3, 1, 1)
        convRelu(3, 128, 128, 3, 1, 1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 12*6
        convRelu(4, 128, 256, 3, 1, 1, True)
        convRelu(5, 256, 256, 3, 1, 1)
        convRelu(6, 256, 256, 3, 1, 1)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2)))  # 6*3
        convRelu(7, 256, 512, 3, 1, 1, True)
        convRelu(8, 512, 512, 3, 1, 1)
        convRelu(9, 512, 512, 3, 1, 1)

        self.cnn = cnn
        self.fc6 = FC(512 * 6 * 3, 128)
        self.score_fc = FC(128, 10)
        self.loss = None
        self.cls_prob = None

    def forward(self, im_data, labels=None):
        im_data = network.np_to_variable(im_data, is_cuda=True)
        x = self.cnn(im_data)
        x = x.view(x.size()[0], -1)
        y = self.fc6(x)
        y = F.dropout(y, training=self.training)
        y = self.score_fc(y)
        if self.training:
            self.loss = F.cross_entropy(y, network.np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor))
        self.cls_prob = F.softmax(y)
        return self.cls_prob

    def get_image_blob(self, im):
        
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= np.array([104.00698793])

        im_shape = im_orig.shape
        im_scale_x = float(cfg.TEST.WIDTH) / float(im_shape[1])
        im_scale_y = float(cfg.TEST.HEIGHT) / float(im_shape[0])

        processed_ims = []
        im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        processed_ims.append(im)

        # Create a blob to hold the input images
        blob = self._im_list_to_blob(processed_ims)

        return blob

    def _im_list_to_blob(self, ims):
        img_shape = ims[0].shape   
        num_images = len(ims)
        blob = np.zeros((num_images, img_shape[0], img_shape[1], 1),    
                        dtype=np.float32)           #[nums, h, w, 3]
        for i in xrange(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im[:, :, np.newaxis]
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob

def train(net, optimizer, lr):
    print 'starting training'
    start_step = 0
    end_step = 15000
    lr_decay_steps = {15000}
    lr_decay = 1./10

    train_loss = 0
    step_cnt = 0
    re_cnt = False
    t = Timer()
    t.tic()
    for step in range(start_step, end_step + 1):

        # get one batch
        blobs = data_layer.forward()
        im_data = blobs['data']
        labels = blobs['labels']

        # forward
        net(im_data, labels)
        loss = net.loss
        train_loss += loss.data[0]
        step_cnt += 1

        # backward
        optimizer.zero_grad()
        loss.backward()
        network.clip_gradient(net, 10.)
        optimizer.step()

        if step % disp_interval == 0:
            duration = t.toc(average=False)
            fps = step_cnt / duration

            log_text = 'step %d, %s, loss: %.4f, fps: %.2f (%.2fs/batch), lr: %.3e' % (
                step, str(datetime.now())[5:], train_loss / step_cnt, fps, 1./fps, lr)
            log_print(log_text, color='green', attrs=['bold'])
            logging.info(log_text)

            log_value('loss', train_loss / step_cnt, step)

            re_cnt = True

        if (step % snap_interval == 0) and step > 0:
            save_name = os.path.join(output_dir, 'mnist_{}.h5'.format(step))
            network.save_net(save_name, net)
            print('save model: {}'.format(save_name))
        if step in lr_decay_steps:
            lr *= lr_decay
            optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

        if re_cnt:
            train_loss = 0
            step_cnt = 0
            t.tic()
            re_cnt = False


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'experiments', 'logs')
    filename = 'mnist_train_' + time.strftime("%Y%m%d-%H%M%S") + '.log'
    configure("runs/phone")

    imdb_name = 'mnist_train'
    cfg_file = 'experiments/cfgs/train_mnist.yml'
    output_dir = 'output/mnist_train'

    rand_seed = 1024
    _DEBUG = True
    exp_name = None # the previous experiment name in TensorBoard

    if rand_seed is not None:
        np.random.seed(rand_seed)

    # load config
    cfg_from_file(cfg_file)
    print 'Loading cfg file from {}'.format(cfg_file)
    print 'Using config:'
    pprint.pprint(cfg)

    lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    disp_interval = cfg.TRAIN.DISPLAY
    log_interval = cfg.TRAIN.LOG_IMAGE_ITERS
    snap_interval = cfg.TRAIN.SNAPSHOT_ITERS

    # load data
    # imdb = get_imdb(imdb_name)
    imdb = get_imdb(imdb_name, os.path.join(cfg.DATA_DIR, 'express', 'pretrain_mnist'), ratio=0.8)
    prepare_roidb(imdb)
    roidb = imdb.roidb
    print 'roidb length: {}'.format(len(roidb))
    data_layer = MnistDataLayer(roidb)

    # load net
    print 'init net'
    net = Net()
    network.weights_normal_init(net)

    net.cuda()
    net.train()
    params = list(net.parameters())
    # optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    optimizer = torch.optim.Adam(params, lr=lr)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    train(net, optimizer, lr)