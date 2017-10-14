# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import yaml
import numpy as np
import numpy.random as npr
import pdb

from utils.cython_bbox import bbox_overlaps, bbox_intersections

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete

DEBUG = False


def proposal_target_layer(rpn_rois, gt_boxes, im_inds, _num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    Parameters
    ----------
    rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
    gt_ishard: (G, 1) {0 | 1} 1 indicates hard
    dontcare_areas: (D, 4) [ x1, y1, x2, y2]
    _num_classes
    ----------
    Returns
    ----------
    rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
    labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
    bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
    bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
    bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    num_images = cfg.TRAIN.IMS_PER_BATCH
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    all_rois = np.vstack(
            (all_rois, np.hstack((im_inds, gt_boxes[:, :4])))
        )

    labels_all = []
    rois_all = []
    bbox_targets_all = []
    bbox_inside_weights_all = []

    for batch_image in range(num_images):
        inds_i = np.where(all_rois[:,0] == batch_image)[0]
        all_rois_i = all_rois[inds_i,:]
        inds_i = np.where(im_inds == batch_image)[0]
        gt_boxes_i = gt_boxes[inds_i,:]


        # Sample rois with classification labels and bounding box regression
        # targets
        labels, rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois_i, gt_boxes_i, fg_rois_per_image,
            rois_per_image, _num_classes)

        _count = 1
        if DEBUG:
            if _count == 1:
                _fg_num, _bg_num = 0, 0
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            _count += 1
            _fg_num += (labels > 0).sum()
            _bg_num += (labels == 0).sum()
            print 'num fg avg: {}'.format(_fg_num / _count)
            print 'num bg avg: {}'.format(_bg_num / _count)
            print 'ratio: {:.3f}'.format(float(_fg_num) / float(_bg_num))

        if labels_all == [] and labels != []:
            labels = labels.reshape(len(labels),1)
            labels_all = labels
            rois_all = rois
            bbox_targets_all = bbox_targets
            bbox_inside_weights_all = bbox_inside_weights
        else:
            if labels != []:
                labels = labels.reshape(len(labels),1)
                labels_all = np.vstack((labels_all,labels))
                rois_all = np.vstack((rois_all,rois))
                bbox_targets_all = np.vstack((bbox_targets_all,bbox_targets))
                bbox_inside_weights_all = np.vstack((bbox_inside_weights_all,bbox_inside_weights))

    rois_all = rois_all.reshape(-1, 5)
    labels_all = labels_all.reshape(-1, 1)
    bbox_targets_all = bbox_targets_all.reshape(-1, _num_classes * 4)
    bbox_inside_weights_all = bbox_inside_weights_all.reshape(-1, _num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights_all > 0).astype(np.float32)

    return rois_all, labels_all, bbox_targets_all, bbox_inside_weights_all, bbox_outside_weights


def _sample_rois(all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: R x G
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)  # R
    max_overlaps = overlaps.max(axis=1)  # R
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = min(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    # Sample background regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    rois = all_rois[keep_inds]

    bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data, num_classes)

    return labels, rois, bbox_targets, bbox_inside_weights


def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                   / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _jitter_gt_boxes(gt_boxes, jitter=0.05):
    """ jitter the gtboxes, before adding them into rois, to be more robust for cls and rgs
    gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
    """
    jittered_boxes = gt_boxes.copy()
    ws = jittered_boxes[:, 2] - jittered_boxes[:, 0] + 1.0
    hs = jittered_boxes[:, 3] - jittered_boxes[:, 1] + 1.0
    width_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * ws
    height_offset = (np.random.rand(jittered_boxes.shape[0]) - 0.5) * jitter * hs
    jittered_boxes[:, 0] += width_offset
    jittered_boxes[:, 2] += width_offset
    jittered_boxes[:, 1] += height_offset
    jittered_boxes[:, 3] += height_offset

    return jittered_boxes
