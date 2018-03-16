import _init_paths
import os
import torch
import cv2
import cPickle
import numpy as np
from datetime import datetime
import pprint
import yaml

import network
from phone_net import PhoneNet
from utils.timer import Timer
import time

from roi_data_layer.multilabel_layer import MultilabelDataLayer, WeightedRandomSampler
from datasets.factory import get_imdb
from tensorboard_logger import configure, log_value
from progressbar import ProgressBar
from termcolor import cprint

def im_detect(net, image_list, height=48, width=256):
    im_data = net.get_image_blob_list(image_list, height=height, width=width)
    cls_prob = net(im_data)
    scores = [cls_prob[i].data.cpu().numpy() for i in range(13)]
    return scores

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

    start_step = 40000
    end_step = 100000
    lr_decay_steps = [70000, 100000]
    ohem_steps = a = range(40000, 95000, 5000)
    lr_decay = 1./10
    lr = 0.001
    momentum = 0.9
    weight_decay = 0.0005
    disp_interval = 20
    log_interval = 20
    snap_interval = 4000
    width = 256
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
        ohem_steps=ohem_steps,
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
    

    # imdb = get_imdb(imdb_name)
    # imdb = get_imdb(imdb_name, os.path.join('data', 'express', 'pretrain_db_benchmark'), ratio=0.8)
    imdb = get_imdb(imdb_name, os.path.join('data', 'express', 'pretrain_db_benchmark_extra'), ratio=0.8)
    prepare_roidb(imdb)
    roidb = imdb.roidb
    print 'roidb length: {}'.format(len(roidb))

    weight = np.ones((len(roidb),)).astype(np.float64)
    # sampler = RandomSampler(len(roidb), batch_size)
    sampler = WeightedRandomSampler(len(roidb), batch_size, weight)
    data_layer = MultilabelDataLayer(roidb, 12, sampler, batch_size=batch_size, height=height, width=width)


    print 'init net'
    net = PhoneNet(classes=imdb.classes, bn=False)
    network.weights_normal_init(net)
    # model_file = 'output/mnist_out/mnist_4200_no_bn.h5'
    # network.load_pretrain_net(model_file, net, num=17*2)
    model_file = 'output/phone_train/TIME-20171229-205253/phone_40000.h5'
    network.load_net(model_file, net)
    net.cuda()
    net.train()
    params = list(net.parameters())[4*2:]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    print 'starting training'
    train_loss = 0
    step_cnt = 0
    re_cnt = False
    timestamp_start = datetime.now()
    timestamp_this = datetime.now()
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
            duration = (datetime.now() - timestamp_this).total_seconds()
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

        if step in ohem_steps:
            ohem_path = os.path.join(log_out, 'ohem_{}'.format(step))
            if not os.path.exists(ohem_path):
                os.makedirs(ohem_path)
            cache_file = os.path.join(ohem_path, 'detection_score.pkl')
            num_images = len(imdb.image_index)
            all_boxes = [[[] for _ in xrange(13)]
                         for _ in xrange(num_images)]

            net.eval()
            print 'starting OHEM ...'
            pbar = ProgressBar(maxval=num_images)
            pbar.start()
            test_batch_size=128
            for i in range(0, num_images, test_batch_size):
                im_list = []
                for k in range(min(test_batch_size, num_images-i)):
                    im = cv2.imread(imdb.image_path_at(i+k))
                    x, y, w, h = roidb[i+k]['bbox']
                    im = im[y:y+h, x:x+w, :]
                    im_list.append(im)
                scores = im_detect(net, im_list, height=48, width=256)
                for k in range(min(test_batch_size, num_images-i)):
                    for j in range(13):
                        all_boxes[i+k][j] = scores[j][k:k+1]
                pbar.update(i)
            pbar.finish()
            with open(cache_file, 'wb') as fid:
                cPickle.dump(all_boxes, fid, cPickle.HIGHEST_PROTOCOL)
            print 'Evaluating detections'
            weight = imdb.evaluate_ohem(all_boxes, ohem_path, roidb)
            sampler.set_weight(weight)
            del all_boxes

            net.train()

        if re_cnt:
            train_loss = 0
            step_cnt = 0
            timestamp_this = datetime.now()
            re_cnt = False

