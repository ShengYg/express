import _init_paths
import os
import torch
import numpy as np
import random
from datetime import datetime
import pprint
import yaml

import network
from phone_net import PhoneNet
from utils.timer import Timer
import time

from roi_data_layer.multilabel_layer import MultilabelDataLayer
from datasets.factory import get_imdb
from tensorboard_logger import configure, log_value


from termcolor import cprint

def get_log_dir(cfg):
    now = datetime.now()
    name = 'TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('output', 'phone_train', name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir

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
    configure("runs/phone")

    imdb_name = 'phone_train'

    start_step = 0
    end_step = 60000
    lr_decay_steps = [50000, 60000]
    lr_decay = 1./10
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    disp_interval = 20
    log_interval = 20
    snap_interval = 4000
    width = 240
    height = 48
    batch_size = 128
    iter_size = 1

    cfg = dict(
        batch_size=batch_size,
        disp_interval=disp_interval,
        end_step=end_step,
        height=height,
        width=width,
        iter_size=iter_size,
        momentum=momentum,
        snap_interval=snap_interval,
        start_step=start_step,
        weight_decay=weight_decay,
        log_interval=log_interval,
        lr=lr,
        lr_decay_steps=lr_decay_steps,
    )
    log_headers = [
        'iteration',
        'lr',
        'train/loss',
        'train/loss1',
        'train/loss2',
        'train/loss3',
        'train/loss4',
        'train/loss5',
        'train/loss6',
        'train/loss7',
        'train/loss8',
        'train/loss9',
        'train/loss10',
        'train/loss11',
        'train/loss12',
        'train/losslength',
        'elapsed_time',
    ]
    log_out = get_log_dir(cfg)
    if not os.path.exists(os.path.join(log_out, 'log.csv')):
        with open(os.path.join(log_out, 'log.csv'), 'w') as f:
            f.write(','.join(log_headers) + '\n')

    # load data
    # imdb = get_imdb(imdb_name)
    # imdb = get_imdb(imdb_name, os.path.join('data', 'express', 'pretrain_db_benchmark'), ratio=0.8)
    imdb = get_imdb(imdb_name, os.path.join('data', 'express', 'pretrain_db_benchmark_extra'), ratio=0.8)
    prepare_roidb(imdb)
    roidb = imdb.roidb
    print 'roidb length: {}'.format(len(roidb))
    data_layer = MultilabelDataLayer(roidb, 12, batch=batch_size, height=height, width=width)

    net_bn = False
    net = None
    if net_bn:
        print 'init net'
        net = PhoneNet(classes=imdb.classes, bn=True)
        network.weights_normal_init(net)

        model_file = 'output/mnist_out/mnist_2200.h5'
        network.load_pretrain_net(model_file, net, num=17*6)

        net.cuda()
        net.train()

        params = list(net.parameters())[4*6:]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=0)
    else:
        print 'init net'
        net = PhoneNet(classes=imdb.classes, bn=False)
        network.weights_normal_init(net)

        model_file = 'output/mnist_out/mnist_4200_no_bn.h5'
        network.load_pretrain_net(model_file, net, num=17*2)

        net.cuda()
        net.train()

        params = list(net.parameters())[4*2:]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    print 'starting training'
    train_loss = 0
    step_cnt = 0
    re_cnt = False
    t = Timer()
    t.tic()
    timestamp_start = datetime.now()
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

            with open(os.path.join(log_out, 'log.csv'), 'a') as f:
                log = [step, lr, train_loss / step_cnt]
                log += [net.out_loss[0].data.cpu().numpy()[0]]
                log += [net.out_loss[1].data.cpu().numpy()[0]]
                log += [net.out_loss[2].data.cpu().numpy()[0]]
                log += [net.out_loss[3].data.cpu().numpy()[0]]
                log += [net.out_loss[4].data.cpu().numpy()[0]]
                log += [net.out_loss[5].data.cpu().numpy()[0]]
                log += [net.out_loss[6].data.cpu().numpy()[0]]
                log += [net.out_loss[7].data.cpu().numpy()[0]]
                log += [net.out_loss[8].data.cpu().numpy()[0]]
                log += [net.out_loss[9].data.cpu().numpy()[0]]
                log += [net.out_loss[10].data.cpu().numpy()[0]]
                log += [net.out_loss[11].data.cpu().numpy()[0]]
                log += [net.out_loss[12].data.cpu().numpy()[0]]
                log += [(datetime.now() - timestamp_start).total_seconds()]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            re_cnt = True

        if (step % snap_interval == 0) and step > 0:
            save_name = os.path.join(log_out, 'phone_{}.h5'.format(step))
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

