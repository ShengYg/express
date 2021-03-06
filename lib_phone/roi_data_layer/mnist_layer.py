from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
import yaml
import os
import random
import cPickle
import cv2


class MnistDataLayer(object):

    def __init__(self, roidb, num_labels=1):
        self._roidb = roidb
        self._num_labels = num_labels
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
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
        
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return self._get_minibatch(minibatch_db, self._num_labels)

    def _get_minibatch(self, roidb, num_labels):
        num_images = len(roidb)

        im_blob = self._get_image_blob(roidb)
        # im_blob: batch_size * channel(3) * height * width

        blobs = {'data': im_blob}
        all_labels = np.ones((cfg.TRAIN.IMS_PER_BATCH))
        all_labels[:] = 10
        for i in range(num_images):
            all_labels[i] = roidb[i]['labels']
        blobs['labels'] = all_labels.astype(np.int32)
        return blobs

    def _get_image_blob(self, roidb):

        num_images = len(roidb)
        processed_ims = []
        for i in xrange(num_images):
            im = cv2.imread(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            im = self._prep_im_for_blob(im, cfg.PIXEL_MEANS)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = self._im_list_to_blob(processed_ims)
        # blob: batch_size(3) * channel(3) * height * width

        return blob

    def _im_list_to_blob(self, ims):
        img_shape = ims[0].shape   
        num_images = len(ims)   # 3
        blob = np.zeros((num_images, img_shape[0], img_shape[1], 3),    
                        dtype=np.float32)           #[nums, h, w, 3]
        for i in xrange(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        # Move channels (axis 3) to axis 1
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob

    def _prep_im_for_blob(self, im, pixel_means):
        """Mean subtract and scale an image for use in a blob."""
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape
        im_scale_x = float(cfg.TRAIN.WIDTH) / float(im_shape[1])
        im_scale_y = float(cfg.TRAIN.HEIGHT) / float(im_shape[0])

        
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        return im

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs