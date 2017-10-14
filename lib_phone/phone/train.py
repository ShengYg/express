# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

import caffe
from fast_rcnn.config import cfg
from utils.timer import Timer
import numpy as np
import os

from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format
from tensorboard_logger import configure, log_value

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, solver_prototxt, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir

        self.solver = caffe.SGDSolver(solver_prototxt)
        if pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(pretrained_model)
            self.solver.net.copy_from(pretrained_model)

        self.solver_param = caffe_pb2.SolverParameter()
        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def snapshot(self):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.solver.net

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (self.solver_param.snapshot_prefix + infix +
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)
        
        return filename

    def train_model(self, max_iters):
        """Network training loop."""
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()

            loss_all =  self.solver.net.blobs['loss0'].data.copy() + \
                        self.solver.net.blobs['loss1'].data.copy() + \
                        self.solver.net.blobs['loss2'].data.copy() + \
                        self.solver.net.blobs['loss3'].data.copy() + \
                        self.solver.net.blobs['loss4'].data.copy() + \
                        self.solver.net.blobs['loss5'].data.copy() + \
                        self.solver.net.blobs['loss6'].data.copy() + \
                        self.solver.net.blobs['loss7'].data.copy() + \
                        self.solver.net.blobs['loss8'].data.copy() + \
                        self.solver.net.blobs['loss9'].data.copy() + \
                        self.solver.net.blobs['loss10'].data.copy() + \
                        self.solver.net.blobs['loss11'].data.copy() + \
                        self.solver.net.blobs['loss_length'].data.copy()
            log_value('loss1', self.solver.net.blobs['loss0'].data.copy(), self.solver.iter)
            log_value('loss2', self.solver.net.blobs['loss1'].data.copy(), self.solver.iter)
            log_value('loss3', self.solver.net.blobs['loss2'].data.copy(), self.solver.iter)
            log_value('loss4', self.solver.net.blobs['loss3'].data.copy(), self.solver.iter)
            log_value('loss5', self.solver.net.blobs['loss4'].data.copy(), self.solver.iter)
            log_value('loss6', self.solver.net.blobs['loss5'].data.copy(), self.solver.iter)
            log_value('loss7', self.solver.net.blobs['loss6'].data.copy(), self.solver.iter)
            log_value('loss8', self.solver.net.blobs['loss7'].data.copy(), self.solver.iter)
            log_value('loss9', self.solver.net.blobs['loss8'].data.copy(), self.solver.iter)
            log_value('loss10', self.solver.net.blobs['loss9'].data.copy(), self.solver.iter)
            log_value('loss11', self.solver.net.blobs['loss10'].data.copy(), self.solver.iter)
            log_value('loss12', self.solver.net.blobs['loss11'].data.copy(), self.solver.iter)
            log_value('loss_length', self.solver.net.blobs['loss_length'].data.copy(), self.solver.iter)
            log_value('loss_all', loss_all, self.solver.iter)

            if self.solver.iter % (10 * self.solver_param.display) == 0:    #200
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:            #10000
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths

def prepare_roidb(imdb):
    roidb = imdb.roidb
    for i in xrange(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)
        

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:   #true
        print 'prepare'
        imdb.roidb
        # print 'Appending horizontally-flipped training examples...'
        # imdb.append_flipped_images()
        print 'done'

    print 'Preparing training data...'
    prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def train_net(solver_prototxt, roidb, output_dir, pretrained_model=None, max_iters=40000):

    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths
