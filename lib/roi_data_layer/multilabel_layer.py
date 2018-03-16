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

def precision_cut(weight):
    weight = weight.astype(np.float64)
    weight = weight / np.sum(weight)
    cut = weight.shape[0]-1
    while np.sum(weight[:cut]) >= 1.0:
        cut -= 1
    print '#######precision cut, so cut = {}-{}'.format(weight.shape[0], weight.shape[0]-cut-1)
    return weight[:cut+1]

class RandomSampler(object):
    def __init__(self, length, batch_size):
        self._length = length
        self._batch_size = batch_size
        _shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        self._perm = np.random.permutation(np.arange(self._length))
        self._cur = 0

    def forward(self):
        if self._cur + self._batch_size >= self._length:
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._batch_size]
        self._cur += self._batch_size
        return db_inds

class WeightedRandomSampler(object):
    def __init__(self, length, batch_size, weight):
        self._length = length
        self._batch_size = batch_size
        self._weight = precision_cut(weight)
        self._batch = 1000
        self._shuffle_roidb_inds()

    def set_weight(self, weight):
        self._weight = precision_cut(weight)
        self._shuffle_roidb_inds()

    def _shuffle_roidb_inds(self):
        print 'reshuffle......it means {}*{} batches or new weights'.format(self._batch, self._batch_size)
        choose = npr.multinomial(self._batch * self._batch_size, self._weight)
        self._perm = npr.permutation(np.repeat(np.arange(self._length), choose))
        self._cur = 0

    def forward(self):
        if self._cur + self._batch_size >= self._batch * self._batch_size:
            self._shuffle_roidb_inds()

        db_inds = self._perm[self._cur:self._cur + self._batch_size]
        self._cur += self._batch_size
        return db_inds

class MultilabelDataLayer(object):
    def __init__(self, roidb, num_labels, sampler, batch_size=128, height=48, width=256):
        self._roidb = roidb
        self._num_labels = num_labels
        self._height = height
        self._width = width
        self._batch_size = batch_size
        self._sampler = sampler

    def _get_next_minibatch(self):
        
        db_inds = self._sampler.forward()
        minibatch_db = [self._roidb[i] for i in db_inds]
        return self._get_minibatch(minibatch_db, self._num_labels)

    def _get_minibatch(self, roidb, num_labels):
        num_images = len(roidb)
        im_blob = self._get_image_blob(roidb)
        # im_blob: batch_size * channel(3) * height * width

        blobs = {'data': im_blob}
        all_labels = np.ones((self._batch_size, num_labels))
        all_labels[:] = 10
        all_length = np.zeros((self._batch_size))
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
            im = self._prep_im_for_blob(im, cfg.PIXEL_MEANS, roidb[i]['bbox'])
            processed_ims.append(im)

        blob = self._im_list_to_blob(processed_ims)
        # blob: batch_size(3) * channel(3) * height * width

        return blob

    def _im_list_to_blob(self, ims):
        img_shape = ims[0].shape   
        num_images = len(ims)
        blob = np.zeros((num_images, self._height, self._width, 3), dtype=np.float32)
        for i in xrange(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
            # blob[i, self._height/2-im.shape[0]/2:self._height/2+(im.shape[0]+1)/2, 
            #         self._width/2-im.shape[1]/2:self._width/2+(im.shape[1]+1)/2, :] = im
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob

    def _prep_im_for_blob(self, im, pixel_means, bbox):
        """Mean subtract and scale an image for use in a blob."""
        im = im.astype(np.float32, copy=False)
        im -= pixel_means
        im_shape = im.shape

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

        # v1
        im_scale_x = float(self._width) / float(crop_w)
        im_scale_y = float(self._height ) / float(crop_h)
        crop_img = cv2.resize(crop_img, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        # # v2
        # im_scale = float(self._width) / float(crop_w)
        # if im_scale * crop_h > self._height:
        #     im_scale = float(self._height ) / float(crop_h)
        # crop_img = cv2.resize(crop_img, None, None, fx=im_scale, fy=im_scale,
        #                 interpolation=cv2.INTER_LINEAR)
        return crop_img

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs