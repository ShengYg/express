# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a Fast R-CNN network.

RoIDataLayer implements a Caffe Python layer.
"""

import caffe
from fast_rcnn.config import cfg
from roi_data_layer.minibatch import get_minibatch
import numpy as np
import yaml
from multiprocessing import Process, Queue
import os
import random
import cPickle
import cv2

'''
blobs
data        batch * 3 * height * width
im_info     1 * 3 ==> batch_size(3) * 3
gt_boxes    num * 5 ==> 3 * num * 5     the rest are all 0
'''
# len(input_list) = 8808
class RoIDataLayer(caffe.Layer):

    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in self._roidb])
            heights = np.array([r['height'] for r in self._roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            if inds.shape[0]%2:
                inds = inds[:-1]
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            self._perm = inds
        else:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self): # softmax
        """Return the roidb indices for the next minibatch."""
        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        If cfg.TRAIN.USE_PREFETCH is True, then blobs will be computed in a
        separate process and made available through self._blob_queue.
        """
        
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return get_minibatch(minibatch_db, self._num_classes)   # _num_classes: 21

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()

    def setup(self, bottom, top):
        """Setup the RoIDataLayer."""

        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._num_classes = layer_params['num_classes']

        self._name_to_top_map = {}

        # data blob: holds a batch of N images, each with 3 channels
        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
        self._name_to_top_map['data'] = idx
        idx += 1

        if cfg.TRAIN.HAS_RPN:
            top[idx].reshape(1, 3)
            self._name_to_top_map['im_info'] = idx
            idx += 1

            top[idx].reshape(1, 4)
            self._name_to_top_map['gt_boxes'] = idx
            idx += 1

            top[idx].reshape(1)
            self._name_to_top_map['im_inds'] = idx
            idx += 1

        print 'RoiDataLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


class MnistLayer(caffe.Layer):

    def _shuffle_roidb_inds(self):
        
        self._perm = np.random.permutation(np.arange(len(self._roidb)))
        self._cur = 0

    def _get_next_minibatch_inds(self):

        if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._roidb):
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
        self._cur += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):

        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return self._get_minibatch(minibatch_db, self._num_classes)

    def _get_minibatch(self, roidb, num_classes):
        num_images = len(roidb)
        random_scale_inds = np.random.randint(0, high=len(cfg.TRAIN.SCALES),
                                        size=num_images)
        assert(cfg.TRAIN.BATCH_SIZE % num_images == 0) or (cfg.TRAIN.BATCH_SIZE == -1), \
            'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

        im_blob = self._get_image_blob(roidb, random_scale_inds)

        blobs = {'data': im_blob}

        assert len(roidb) == num_images, "{} batch only".format(num_images)
        
        gt_labels = []
        for i in range(num_images):
            gt_labels.append(roidb[i]['labels'])
        blobs['gt_labels'] = np.array(gt_labels).reshape((num_images ,1))

        return blobs

    def _get_image_blob(self, roidb, scale_inds):
        num_images = len(roidb)
        processed_ims = []
        for i in xrange(num_images):
            im = cv2.imread(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            im = self._prep_im_for_blob(im, cfg.PIXEL_MEANS)
            processed_ims.append(im)

        blob = self._im_list_to_blob(processed_ims)

        return blob

    def _prep_im_for_blob(self, im, pixel_means):
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_scale_x = float(cfg.TRAIN.WIDTH) / im.shape[1]
        im_scale_y = float(cfg.TRAIN.HEIGHT) / im.shape[0]
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)

        return im

    def _im_list_to_blob(self, ims):
        blob = np.zeros((len(ims), ims[0].shape[0], ims[0].shape[1], 3),
                        dtype=np.float32)
        for i in xrange(len(ims)):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob

    def set_roidb(self, roidb):
        """Set the roidb to be used by this layer during training."""
        self._roidb = roidb
        self._shuffle_roidb_inds()

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._num_classes = layer_params['num_classes']
        self._name_to_top_map = {}

        idx = 0
        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3,
            cfg.TRAIN.HEIGHT, cfg.TRAIN.WIDTH)
        self._name_to_top_map['data'] = idx
        idx += 1

        top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 1)
        self._name_to_top_map['gt_labels'] = idx
        idx += 1

        print 'MnistLayer: name_to_top:', self._name_to_top_map
        assert len(top) == len(self._name_to_top_map)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
