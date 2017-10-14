import _init_paths
import os
import torch
import numpy as np
import random
from datetime import datetime
import pprint

import network
from phone_net import PhoneNet
from utils.timer import Timer
import time

from roi_data_layer.multilabel_layer import MultilabelDataLayer
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file
from tensorboard_logger import configure, log_value
import logging
from tensorboardX import SummaryWriter

try:
    from termcolor import cprint
except ImportError:
    cprint = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)

def prepare_roidb(imdb):
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)


if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'experiments', 'logs')
    filename = 'phone_train_' + time.strftime("%Y%m%d-%H%M%S") + '.log'
    logging.basicConfig(level = logging.DEBUG,
                        filename = os.path.join(path, filename),
                        filemode = 'w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO) 
    formatter = logging.Formatter('[%(levelname)-8s] %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
    configure("runs/phone")

    imdb_name = 'phone_train'
    cfg_file = 'experiments/cfgs/train_phone.yml'
    output_dir = 'output/phone_train'

    start_step = 0
    end_step = 60000
    lr_decay_steps = {25000, 40000}
    lr_decay = 1./10

    rand_seed = 1024

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
    # snap_interval = cfg.TRAIN.SNAPSHOT_ITERS
    snap_interval = 4000

    # load data
    # imdb = get_imdb(imdb_name)
    imdb = get_imdb(imdb_name, os.path.join(cfg.DATA_DIR, 'express', 'pretrain_db_benchmark'), ratio=1.0)
    prepare_roidb(imdb)
    roidb = imdb.roidb
    print 'roidb length: {}'.format(len(roidb))
    data_layer = MultilabelDataLayer(roidb, 12)

    net_bn = False
    net = None
    if net_bn:
        print 'init net'
        net = PhoneNet(classes=imdb.classes, bn=True)
        network.weights_normal_init(net)

        model_file = 'output/mnist_train/mnist_2200.h5'
        network.load_pretrain_net(model_file, net, num=17*6)

        net.cuda()
        net.train()

        params = list(net.parameters())[4*6:]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=0)
    else:
        print 'init net'
        net = PhoneNet(classes=imdb.classes, bn=False)
        network.weights_normal_init(net)

        model_file = 'output/mnist_train/mnist_4200_no_bn.h5'
        network.load_pretrain_net(model_file, net, num=17*2)

        net.cuda()
        net.train()

        params = list(net.parameters())[4*2:]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    print 'starting training'
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
        length = blobs['phone_length']

        # forward
        net(im_data, labels, length)
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

            log_value('loss_all', train_loss / step_cnt, step)
            log_value('loss1', net.out_loss[0].data.cpu().numpy()[0], step)
            log_value('loss2', net.out_loss[1].data.cpu().numpy()[0], step)
            log_value('loss3', net.out_loss[2].data.cpu().numpy()[0], step)
            log_value('loss4', net.out_loss[3].data.cpu().numpy()[0], step)
            log_value('loss5', net.out_loss[4].data.cpu().numpy()[0], step)
            log_value('loss6', net.out_loss[5].data.cpu().numpy()[0], step)
            log_value('loss7', net.out_loss[6].data.cpu().numpy()[0], step)
            log_value('loss8', net.out_loss[7].data.cpu().numpy()[0], step)
            log_value('loss9', net.out_loss[8].data.cpu().numpy()[0], step)
            log_value('loss10', net.out_loss[9].data.cpu().numpy()[0], step)
            log_value('loss11', net.out_loss[10].data.cpu().numpy()[0], step)
            log_value('loss12', net.out_loss[11].data.cpu().numpy()[0], step)
            log_value('loss_length', net.out_loss[12].data.cpu().numpy()[0], step)

            re_cnt = True

        if (step % snap_interval == 0) and step > 0:
            save_name = os.path.join(output_dir, 'phone_{}.h5'.format(step))
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

