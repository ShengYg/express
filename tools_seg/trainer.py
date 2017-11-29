import datetime
import math
import os
import shutil

import numpy as np
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
from tensorboard_logger import configure, log_value

from utils import label_accuracy_score
from network import FCN32s, Front_end, Context

def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    # log_p: (n, h, w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())

def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        FCN32s,
        Front_end,
        Context,
        nn.Upsample,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))

class Trainer(object):

    def __init__(self, cuda, model,
                 train_loader, out,
                 size_average=False, cfg=None, pos=False):
        self.cuda = cuda
        self.model = model
        self.train_loader = train_loader
        self.out = out
        if not os.path.exists(self.out):
            os.makedirs(self.out)
        self.size_average = size_average
        self.pos = pos

        self.cfg = cfg
        self.lr = cfg['lr']
        self.max_iter = cfg['max_iteration']
        self.snap_interval = cfg['snap_interval']
        self.log_interval = cfg['log_interval']
        self.lr_decay_interval = cfg['lr_decay_interval']

        # self.optim = torch.optim.SGD(
        #         [
        #             {'params': list(get_parameters(self.model, bias=True))[:13],
        #              'lr': 0, 'weight_decay': 0},
        #             {'params': list(get_parameters(self.model, bias=False))[:13],
        #              'lr': 0, 'weight_decay': 0},
        #             {'params': list(get_parameters(self.model, bias=True))[13:],
        #              'lr': self.lr * 2, 'weight_decay': 0},
        #             {'params': list(get_parameters(self.model, bias=False))[13:]},
        #         ],
        #         lr=self.lr,
        #         momentum=cfg['momentum'],
        #         weight_decay=cfg['weight_decay']
        #     )
        self.optim = torch.optim.SGD(
                [
                    {'params': list(get_parameters(self.model, bias=True)),
                     'lr': self.lr * 2, 'weight_decay': 0},
                    {'params': list(get_parameters(self.model, bias=False))},
                ],
                lr=self.lr,
                momentum=cfg['momentum'],
                weight_decay=cfg['weight_decay']
            )

        self.timestamp_start = \
            datetime.datetime.now()

        self.log_headers = [
            'epoch',
            'iteration',
            'lr',
            'train/loss',
            'train/loss_pos',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not os.path.exists(os.path.join(self.out, 'log.csv')):
            with open(os.path.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.best_mean_iu = 0


    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target, target_pos) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.cuda:
                data, target = data.cuda(), target.cuda()
                if self.pos:
                    target_pos = target_pos.cuda()
            data, target = Variable(data), Variable(target)
            if self.pos:
                target_pos =  Variable(target_pos)
            self.optim.zero_grad()
            score, score_pos = self.model(data)
            loss = cross_entropy2d(score, target, size_average=self.size_average)
            if self.pos:
                loss2 = cross_entropy2d(score_pos, target_pos, size_average=self.size_average)
                loss += loss2
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            if self.iteration % self.log_interval == 0:
                log_value('loss', loss.data[0], self.iteration)
                if self.pos:
                    log_value('loss_pos', loss2.data[0], self.iteration)
                metrics = []
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = target.data.cpu().numpy()
                for lt, lp in zip(lbl_true, lbl_pred):
                    acc, acc_cls, mean_iu, fwavacc = \
                        label_accuracy_score([lt], [lp], n_class=n_class)
                    metrics.append((acc, acc_cls, mean_iu, fwavacc))
                metrics = np.mean(metrics, axis=0)

                with open(os.path.join(self.out, 'log.csv'), 'a') as f:
                    elapsed_time = (
                        datetime.datetime.now() -
                        self.timestamp_start).total_seconds()
                    log = [self.epoch, self.iteration] + [self.lr] + [loss.data[0]]
                    if self.pos:
                        log += [loss2.data[0]]
                    else:
                        log += [None]
                    log += metrics.tolist() + [''] * 5 + [elapsed_time]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

            if (self.iteration % self.snap_interval == 0) and self.iteration > 0:
                save_name = os.path.join(self.out, 'phone_{}.h5'.format(self.iteration))
                save_net(save_name, self.model)

            if self.iteration in self.lr_decay_interval:
                self.lr /= 10
                
                # self.optim = torch.optim.SGD(
                #     [
                #         {'params': list(get_parameters(self.model, bias=True))[:13],
                #          'lr': 0, 'weight_decay': 0},
                #         {'params': list(get_parameters(self.model, bias=False))[:13],
                #          'lr': 0, 'weight_decay': 0},
                #         {'params': list(get_parameters(self.model, bias=True))[13:],
                #          'lr': self.lr * 2, 'weight_decay': 0},
                #         {'params': list(get_parameters(self.model, bias=False))[13:]},
                #     ],
                #     lr=self.lr,
                #     momentum=cfg['momentum'],
                #     weight_decay=cfg['weight_decay']
                # )
                self.optim = torch.optim.SGD(
                    [
                        {'params': list(get_parameters(self.model, bias=True)),
                         'lr': self.lr * 2, 'weight_decay': 0},
                        {'params': list(get_parameters(self.model, bias=False))},
                    ],
                    lr=self.lr,
                    momentum=cfg['momentum'],
                    weight_decay=cfg['weight_decay']
                )

    def train(self):
        print '## Starting training'
        configure("./tools_seg/logs/{}".format(self.out.split('/')[-1]))
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
