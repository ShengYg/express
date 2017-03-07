# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from fast_rcnn.config import cfg
from generate_anchors import generate_anchors_person
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from fast_rcnn.nms_wrapper import nms

DEBUG = False

"""
bottom[0]   rpn_cls_prob    (3, 18, h, w)
bottom[1]   rpn_bbox_pred   (3, 36, h, w)
bottom[2]   im_info         (1, 3) ==> (3, 3)                               3 means (h, w, scale_ratio)
top[0]      rpn_rois        (post_nms_topN, 5) ==> (3, post_nms_topN, 5)    5 means (batch, x1, y1, x2, y2)
#top[1]     scores          (n, post_nms_topN, 1)
"""

class ProposalLayerTrain(caffe.Layer):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    Converts RPN outputs (per-anchor scores and bbox regression estimates)
    into object proposals
    """

    def setup(self, bottom, top):
        # parse the layer parameter string, which must be valid YAML
        layer_params = yaml.load(self.param_str)

        self._feat_stride = layer_params['feat_stride']                     #16
        # anchor_scales = layer_params.get('scales', (8, 16, 32))
        # self._anchors = generate_anchors(scales=np.array(anchor_scales))    #np.(9, 4)
        self._anchors = generate_anchors_person()
        self._num_anchors = self._anchors.shape[0]                          #9

        if DEBUG:
            print 'feat_stride: {}'.format(self._feat_stride)
            print 'anchors:'
            print self._anchors

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        top[0].reshape(1, 5)

        # scores blob: holds scores for R regions of interest
        if len(top) > 1:
            top[1].reshape(1, 1, 1, 1)

    def forward(self, bottom, top):
        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)
        
        batch_size = cfg.TRAIN.IMS_PER_BATCH
        assert bottom[0].data.shape[0] == batch_size, \
            'Only {} item batches are supported'.format(batch_size)

        cfg_key = 'TRAIN'

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N  #12000, 6000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N #2000,  300
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH     #0.7,   0.7
        min_size      = cfg[cfg_key].RPN_MIN_SIZE       #16,    16

        # out0 = np.zeros((batch_size, post_nms_topN, 5), dtype=np.float32)
        # out1 = np.zeros((batch_size, post_nms_topN, 1), dtype=np.float32)

        scores = bottom[0].data[:, self._num_anchors:, :, :]  #(n, 9, h, w)
        bbox_deltas = bottom[1].data                          #(n, 36, h, w)
        im_info = bottom[2].data

        if DEBUG:
            print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
            print 'scale: {}'.format(im_info[2])

        # 1. Generate proposals from bbox deltas and shifted anchors
        height, width = scores.shape[-2:]

        if DEBUG:
            print 'score map size: {}'.format(scores.shape)

        # Enumerate all shifts
        shift_x = np.arange(0, width) * self._feat_stride
        shift_y = np.arange(0, height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()  #(w*h, 4)

        # Enumerate all shifted anchors:
        #
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = self._num_anchors
        K = shifts.shape[0]
        anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        anchors = anchors.reshape((K * A, 4))
        blob_all = []

        for batch_image in range(batch_size):
            # the first set of _num_anchors channels are bg probs
            # the second set are the fg probs, which we want
            scores_i = scores[batch_image]  #(1, 9, h, w)
            scores_i = scores_i.reshape(1,scores_i.shape[0],scores_i.shape[1],scores_i.shape[2])
            scores_i = scores_i.transpose((0, 2, 3, 1)).reshape((-1, 1))

            bbox_deltas_i = bbox_deltas[batch_image]              #(1, 36, h, w)
            bbox_deltas_i = bbox_deltas_i.reshape(1,bbox_deltas_i.shape[0],bbox_deltas_i.shape[1],bbox_deltas_i.shape[2])
            bbox_deltas_i = bbox_deltas_i.transpose((0, 2, 3, 1)).reshape((-1, 4))

            im_info_i = im_info[batch_image]            

            # Convert anchors into proposals(predicted bboxes) via bbox transformations
            proposals = bbox_transform_inv(anchors, bbox_deltas_i)

            # 2. clip predicted boxes to image
            proposals = clip_boxes(proposals, im_info_i[:2])

            # 3. remove predicted boxes with either height or width < threshold
            # (NOTE: convert min_size to input image scale stored in im_info[2])
            keep = _filter_boxes(proposals, min_size * im_info_i[2])
            proposals = proposals[keep, :]
            scores_i = scores_i[keep]

            # 4. sort all (proposal, score) pairs by score from highest to lowest
            # 5. take top pre_nms_topN (e.g. 6000)
            order = scores_i.ravel().argsort()[::-1]
            if pre_nms_topN > 0:
                order = order[:pre_nms_topN]
            proposals = proposals[order, :]
            scores_i = scores_i[order]

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)
            keep = nms(np.hstack((proposals, scores_i)), nms_thresh)
            if post_nms_topN > 0:
                keep = keep[:post_nms_topN]
            proposals = proposals[keep, :]
            scores_i = scores_i[keep]
            if proposals.shape[0] < post_nms_topN:
                proposals = np.vstack((proposals, np.zeros((post_nms_topN - proposals.shape[0], 4), dtype=np.float32)))
                scores_i = np.vstack((scores_i, np.zeros((post_nms_topN - scores_i.shape[0], 1), dtype=np.float32)))

            # Output rois blob
            # Our RPN implementation only supports a single input image, so all
            # batch inds are 0
            batch_inds = np.zeros((post_nms_topN, 1), dtype=np.float32)
            batch_inds.fill(batch_image)
            blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

            if blob_all == []:
                blob_all = blob.copy()
            else:
                blob_all = np.vstack((blob_all,blob))
            # out0[batch_image] = blob
            # if len(top) > 1:
            #     out1[batch_image] = scores_i

        top[0].reshape(*(blob_all.shape))
        top[0].data[...] = blob_all

        # [Optional] output scores blob
        # if len(top) > 1:
        #     top[1].reshape(*(out1.shape))
        #     top[1].data[...] = out1

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
