# --------------------------------------------------------
# multi-label input layer
# --------------------------------------------------------



from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
import yaml
import os
import random
import cPickle
import cv2


class MultilabelDataLayer(object):

    def __init__(self, roidb, num_labels):
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
        all_labels = np.ones((cfg.TRAIN.IMS_PER_BATCH, num_labels))
        all_labels[:] = 10
        all_length = np.zeros((cfg.TRAIN.IMS_PER_BATCH))
        for i in range(num_images):
            labels = roidb[i]['labels']
            length = labels.shape[0]
            all_labels[i, :length] = labels
            all_length[i] = length - 5
        blobs['labels'] = all_labels.astype(np.int32)
        blobs['phone_length'] = all_length.astype(np.int32)
        return blobs

    def _get_image_blob(self, roidb):

        num_images = len(roidb)
        processed_ims = []
        for i in xrange(num_images):
            im = cv2.imread(roidb[i]['image'])
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            # im = self._prep_im_for_blob(im, cfg.PIXEL_MEANS)
            im = self._prep_im_for_blob(im, cfg.PIXEL_MEANS, roidb[i]['bbox'])
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

    def _prep_im_for_blob(self, im, pixel_means, bbox):
        """Mean subtract and scale an image for use in a blob."""
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape

        # im_scale_x = float(cfg.TRAIN.WIDTH) / float(im_shape[1]) * float(25) / float(24)
        # im_scale_y = float(cfg.TRAIN.HEIGHT) / float(im_shape[0]) * float(9) / float(8)
        # im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y,
        #                 interpolation=cv2.INTER_LINEAR)
        # x = np.random.randint(0, cfg.TRAIN.WIDTH / 24.0 + 1)
        # y = np.random.randint(0, cfg.TRAIN.HEIGHT / 8.0 + 1)
        # crop_img = im[y:y+cfg.TRAIN.HEIGHT, x:x+cfg.TRAIN.WIDTH, :]

        # crop version 2
        x, y, w, h = bbox
        crop_img, crop_w, crop_h = None, None, None
        if (x, y, w, h) == (0, 0, im.shape[1]-1, im.shape[0]-1):
            crop_img = im[:,:,:]
            crop_w = w
            crop_h = h
        else:
            # print 'it means you are using random shifted image'
            crop_x = np.random.randint(x)
            crop_w = np.random.randint(x+w, im_shape[1]-1) - crop_x
            crop_y = np.random.randint(y)
            crop_h = np.random.randint(y+h, im_shape[0]-1) - crop_y
            crop_img = im[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w, :]

        im_scale_x = float(cfg.TRAIN.WIDTH) / float(crop_w)
        im_scale_y = float(cfg.TRAIN.HEIGHT) / float(crop_h)
        crop_img = cv2.resize(crop_img, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        return crop_img

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs