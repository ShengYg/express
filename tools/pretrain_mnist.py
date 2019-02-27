## caffe pretrain using mnist
## caffe pretrain faster-rcnn
import _init_paths
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
import datasets.imdb
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import google.protobuf.text_format
import numpy as np
import sys
import cPickle
import os

from utils.timer import Timer

class SolverWrapper(object):
    def __init__(self, solver_prototxt, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.output_dir = output_dir
        self.solver = caffe.SGDSolver(solver_prototxt)
        self.solver_param = caffe_pb2.SolverParameter()

        with open(solver_prototxt, 'rt') as f:
            pb2.text_format.Merge(f.read(), self.solver_param)

        self.solver.net.layers[0].set_roidb(roidb)

    def snapshot(self):
        net = self.solver.net
        filename = (self.solver_param.snapshot_prefix + 
                    '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
        filename = os.path.join(self.output_dir, filename)

        net.save(str(filename))
        print 'Wrote snapshot to: {:s}'.format(filename)
        return filename

    def train_model(self, max_iters):
        last_snapshot_iter = -1
        timer = Timer()
        model_paths = []
        while self.solver.iter < max_iters:
            # Make one SGD update
            timer.tic()
            self.solver.step(1)
            timer.toc()
            if self.solver.iter % (10 * self.solver_param.display) == 0:    #200
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:            #10000
                last_snapshot_iter = self.solver.iter
                model_paths.append(self.snapshot())

        if last_snapshot_iter != self.solver.iter:
            model_paths.append(self.snapshot())
        return model_paths


def train_net(solver_prototxt, roidb, output_dir, pretrained_model=None, max_iters=40000):

    sw = SolverWrapper(solver_prototxt, roidb, output_dir,
                       pretrained_model=pretrained_model)

    print 'Solving...'
    model_paths = sw.train_model(max_iters)
    print 'done solving'
    return model_paths

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:   #true
        print 'prepare'
        imdb.roidb
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'
    def prepare_roidb(imdb):
        roidb = imdb.roidb
        for i in xrange(len(imdb.image_index)):
            roidb[i]['image'] = imdb.image_path_at(i)
    prepare_roidb(imdb)
    return imdb.roidb

if __name__ == '__main__':
    solver = '/home/sy/code/re_id/express/models/mnist/VGG16/solver.prototxt'
    max_iters = 20000
    imdb_name = 'mnist_train'
    cfg_file = '/home/sy/code/re_id/express/experiments/cfgs/train_mnist.yml'
    cfg_from_file(cfg_file)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(0)

    imdb = get_imdb(imdb_name, os.path.join('/home/sy/code/re_id/express/data', 'express', 'pretrain_mnist'), ratio=0.8)
    print 'Loaded dataset {} for training'.format(imdb.name)
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb)
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(solver, roidb, output_dir, max_iters=max_iters)