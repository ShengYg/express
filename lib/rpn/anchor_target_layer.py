# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import caffe
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors_express
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform


'''
bottom
rpn_cls_score   batch_size * (2(bg/fg) * 9(anchors)) * conv_size1 * conc_size2
gt_boxes        box_num * 5 ==> 3 * max(box_num) * 5, the rest are all 0
im_info         1 * 3 ==> 3 * 3
data            3 * 3 * h * w

top
rpn_labels                  (3, 1, 9 * height, width)
rpn_bbox_targets            (3, 9 * 4, height, width)
rpn_bbox_inside_weights     (3, 9 * 4, height, width)
rpn_bbox_outside_weights    (3, 9 * 4, height, width)
'''

DEBUG = False
class AnchorTargetLayer(caffe.Layer):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._anchors = generate_anchors_express()
        self._num_anchors = self._anchors.shape[0]      # 9
        self._feat_stride = layer_params['feat_stride'] # 16
        batch_size = cfg.TRAIN.IMS_PER_BATCH

        if DEBUG:
            print 'anchors:'
            print self._anchors
            print 'anchor shapes:'
            print np.hstack((
                self._anchors[:, 2::4] - self._anchors[:, 0::4],
                self._anchors[:, 3::4] - self._anchors[:, 1::4],
            ))
            self._counts = cfg.EPS
            self._sums = np.zeros((1, 4))
            self._squared_sums = np.zeros((1, 4))
            self._fg_sum = 0
            self._bg_sum = 0
            self._count = 0

        # allow boxes to sit over the edge by a small amount
        self._allowed_border = layer_params.get('allowed_border', 0)

        height, width = bottom[0].data.shape[-2:]
        if DEBUG:
            print 'AnchorTargetLayer: height', height, 'width', width

        A = self._num_anchors
        # labels
        top[0].reshape(batch_size, 1, A * height, width)
        # bbox_targets
        top[1].reshape(batch_size, A * 4, height, width)
        # bbox_inside_weights
        top[2].reshape(batch_size, A * 4, height, width)
        # bbox_outside_weights
        top[3].reshape(batch_size, A * 4, height, width)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors
        # measure GT overlap

        # map of shape (..., H, W)
        batch_size = cfg.TRAIN.IMS_PER_BATCH
        height, width = bottom[0].data.shape[-2:]
        gt_boxes = bottom[1].data
        im_info = bottom[2].data
        im_inds = bottom[4].data

        if DEBUG:
            print ''
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])
            print 'height, width: ({}, {})'.format(height, width)

        # 1. Generate proposals from bbox deltas and shifted anchors
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()  #(w*h, 4)
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors   # 9
        K = shifts.shape[0]     # w*h
        all_anchors = (self._anchors.reshape((1, A, 4)) +
                       shifts.reshape((1, K, 4)).transpose((1, 0, 2))) 
        print self._anchors
        print all_anchors
        all_anchors = all_anchors.reshape((K * A, 4))
        total_anchors = int(K * A)

        labels_all = []
        bbox_targets_all = []
        bbox_inside_weights_all = []
        bbox_outside_weights_all = []
        assert bottom[0].data.shape[0] == batch_size, \
            'Only {} item batches are supported'.format(batch_size)

        for batch_image in range(batch_size):
            im_info_i = im_info[batch_image]
            # only keep anchors inside the image
            inds_inside = np.where(
                (all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < im_info_i[1] + self._allowed_border) &  # width
                (all_anchors[:, 3] < im_info_i[0] + self._allowed_border)    # height
            )[0]

            if DEBUG:
                print 'total_anchors', total_anchors
                print 'inds_inside', len(inds_inside)

            # keep only inside anchors
            anchors = all_anchors[inds_inside, :]       # anchor_num * 4
            if DEBUG:
                print 'anchors.shape', anchors.shape

            # label: 1 is positive, 0 is negative, -1 is dont care
            labels = np.empty((len(inds_inside), ), dtype=np.float32)
            labels.fill(-1)

            curr_im_inds = np.where(im_inds == batch_image)[0]
            curr_gt_boxes = gt_boxes[curr_im_inds,:]
            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt)
            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(curr_gt_boxes, dtype=np.float))     # anchor_num * gt_num
            
            argmax_overlaps = overlaps.argmax(axis=1)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]   # anchor_num * 1
            gt_argmax_overlaps = overlaps.argmax(axis=0)
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                       np.arange(overlaps.shape[1])]    # 1 * gt_num
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]   #in order

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps] = 1

            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

            # subsample positive labels if we have too many
            num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
            fg_inds = np.where(labels == 1)[0]
            if len(fg_inds) > num_fg:
                disable_inds = npr.choice(
                    fg_inds, size=(len(fg_inds) - num_fg), replace=False)
                labels[disable_inds] = -1

            # subsample negative labels if we have too many
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
            bg_inds = np.where(labels == 0)[0]
            if len(bg_inds) > num_bg:
                disable_inds = npr.choice(
                    bg_inds, size=(len(bg_inds) - num_bg), replace=False)
                labels[disable_inds] = -1
                #print "was %s inds, disabling %s, now %s inds" % (
                    #len(bg_inds), len(disable_inds), np.sum(labels == 0))

            bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
            bbox_targets = _compute_targets(anchors, curr_gt_boxes[argmax_overlaps, :])

            bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
            bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

            bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
            
            if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
                # uniform weighting of examples (given non-uniform sampling)
                num_examples = np.sum(labels >= 0)
                positive_weights = np.ones((1, 4)) * 1.0 / num_examples
                negative_weights = np.ones((1, 4)) * 1.0 / num_examples
            else:
                assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                        (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
                positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                                    np.sum(labels == 1))
                negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                                    np.sum(labels == 0))
            bbox_outside_weights[labels == 1, :] = positive_weights
            bbox_outside_weights[labels == 0, :] = negative_weights

            if DEBUG:
                self._sums += bbox_targets[labels == 1, :].sum(axis=0)
                self._squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
                self._counts += np.sum(labels == 1)
                means = self._sums / self._counts
                stds = np.sqrt(self._squared_sums / self._counts - means ** 2)
                print 'means:'
                print means
                print 'stdevs:'
                print stds

            # map up to original set of anchors
            labels = _unmap(labels, total_anchors, inds_inside, fill=-1)    # (num, )
            bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)     # (num, 4)
            bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
            bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

            if DEBUG:
                print 'rpn: max max_overlap', np.max(max_overlaps)
                print 'rpn: num_positive', np.sum(labels == 1)
                print 'rpn: num_negative', np.sum(labels == 0)
                self._fg_sum += np.sum(labels == 1)
                self._bg_sum += np.sum(labels == 0)
                self._count += 1
                print 'rpn: num_positive avg', self._fg_sum / self._count
                print 'rpn: num_negative avg', self._bg_sum / self._count

            if labels_all == [] and labels != []:
                labels_all = labels   
                bbox_targets_all = bbox_targets
                bbox_inside_weights_all = bbox_inside_weights
                bbox_outside_weights_all = bbox_outside_weights
            else:
                #print 'Dimensionality check here!';
                if labels != []:
                    labels_all = np.vstack((labels_all,labels))
                    bbox_targets_all = np.vstack((bbox_targets_all,bbox_targets))
                    bbox_inside_weights_all = np.vstack((bbox_inside_weights_all,bbox_inside_weights))
                    bbox_outside_weights_all = np.vstack((bbox_outside_weights_all,bbox_outside_weights))


        # labels
        labels_all = labels_all.reshape((batch_size, height, width, A)).transpose(0, 3, 1, 2)
        labels_all = labels_all.reshape((batch_size, 1, A * height, width))
        top[0].reshape(*labels_all.shape)
        top[0].data[...] = labels_all

        # bbox_targets
        bbox_targets_all = bbox_targets_all \
            .reshape((batch_size, height, width, A * 4)).transpose(0, 3, 1, 2)
        top[1].reshape(*bbox_targets_all.shape)
        top[1].data[...] = bbox_targets_all

        # bbox_inside_weights
        bbox_inside_weights_all = bbox_inside_weights_all \
            .reshape((batch_size, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_inside_weights_all.shape[2] == height
        assert bbox_inside_weights_all.shape[3] == width
        top[2].reshape(*bbox_inside_weights_all.shape)
        top[2].data[...] = bbox_inside_weights_all

        # bbox_outside_weights
        bbox_outside_weights_all = bbox_outside_weights_all \
            .reshape((batch_size, height, width, A * 4)).transpose(0, 3, 1, 2)
        assert bbox_outside_weights_all.shape[2] == height
        assert bbox_outside_weights_all.shape[3] == width
        top[3].reshape(*bbox_outside_weights_all.shape)
        top[3].data[...] = bbox_outside_weights_all


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] >= 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
