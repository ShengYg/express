# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr
import cv2
from fast_rcnn.config import cfg
from utils.blob import prep_im_for_blob, im_list_to_blob

def get_minibatch(roidb, num_classes):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)     #3
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images) #[0, 0, 0]
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))

    # Get the input image blob, formatted for caffe
    im_blob, im_scales, im_info = _get_image_blob(roidb, random_scale_inds)
    # im_blob: batch_size(3) * channel(3) * height * width

    blobs = {'data': im_blob}

    assert len(im_scales) == num_images, "{} batch only".format(num_images)
    assert len(roidb) == num_images, "{} batch only".format(num_images)
    # gt boxes: (x1, y1, x2, y2, cls)
    # blobs['gt_boxes'] = []
    
    gt_boxes_all = []
    im_inds_all = []
    im_info_all = []
    for i in range(num_images):
        gt_inds = np.where(roidb[i]['gt_classes'] != 0)[0]
        gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
        im_inds = np.empty((len(gt_inds), 1), dtype=np.float32)

        gt_boxes[:, 0:4] = roidb[i]['boxes'][gt_inds, :] * im_scales[i]
        gt_boxes[:, 4] = roidb[i]['gt_classes'][gt_inds]
        im_inds[:] = i
        if 'gt_pids' in roidb[i]:
            gt_boxes = np.hstack(
                [gt_boxes, roidb[i]['gt_pids'][gt_inds, np.newaxis]])
        if gt_boxes_all == []:
            gt_boxes_all = gt_boxes
            im_inds_all = im_inds
            im_info_all = np.array([[im_info[i][0], im_info[i][1], im_scales[i]]],dtype=np.float32)
        else:
            gt_boxes_all = np.vstack((gt_boxes_all, gt_boxes))
            im_inds_all = np.vstack((im_inds_all, im_inds))
            im_info_all = np.vstack((im_info_all, [[im_info[i][0], im_info[i][1], im_scales[i]]]))
    blobs['gt_boxes'] = gt_boxes_all
    blobs['im_inds'] = im_inds_all
    blobs['im_info'] = im_info_all

    return blobs

def _sample_rois(roidb, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # label = class RoI has max overlap with
    labels = roidb['max_classes']
    overlaps = roidb['max_overlaps']
    rois = roidb['boxes']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
                fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(bg_rois_per_this_image,
                                        bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
                bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[fg_rois_per_this_image:] = 0
    overlaps = overlaps[keep_inds]
    rois = rois[keep_inds]
    # Auxiliary label if available
    aux_label = None
    bg_aux_label = 5532
    if 'gt_pids' in roidb:
        aux_label = roidb['gt_pids']
        aux_label = aux_label[keep_inds]
        aux_label[fg_rois_per_this_image:] = bg_aux_label
        inds = np.where(aux_label == -1)
        aux_label[inds] = bg_aux_label

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            roidb['bbox_targets'][keep_inds, :], num_classes)
    
    return labels, overlaps, rois, bbox_targets, bbox_inside_weights, aux_label

def _get_image_blob(roidb, scale_inds):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    processed_ims = []
    im_scales = []
    im_info = []
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]   # 600
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        im_info.append([im.shape[0], im.shape[1]])
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)
    # blob: batch_size(3) * channel(3) * height * width

    return blob, im_scales, im_info

def _project_im_rois(im_rois, im_scale_factor):
    """Project image RoIs into the rescaled training image."""
    rois = im_rois * im_scale_factor
    return rois

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
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

def _vis_minibatch(im_blob, rois_blob, labels_blob, overlaps):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    for i in xrange(rois_blob.shape[0]):
        rois = rois_blob[i, :]
        im_ind = rois[0]
        roi = rois[1:]
        im = im_blob[im_ind, :, :, :].transpose((1, 2, 0)).copy()
        im += cfg.PIXEL_MEANS
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        cls = labels_blob[i]
        plt.imshow(im)
        print 'class: ', cls, ' overlap: ', overlaps[i]
        plt.gca().add_patch(
            plt.Rectangle((roi[0], roi[1]), roi[2] - roi[0],
                          roi[3] - roi[1], fill=False,
                          edgecolor='r', linewidth=3)
            )
        plt.show()

# new func
def _vis_minibatch_(im_blob, gt_boxes):
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt

    im = im_blob.transpose(1, 2, 0).copy()
    im += cfg.PIXEL_MEANS
    im = im[:, :, ::-1].astype(np.uint8)

    plt.imshow(im)
    for x1, y1, x2, y2, is_person, pid in gt_boxes:
        assert is_person == 1
        ec = 'r' if pid == -1 else 'g'
        if pid != -1:
            plt.gca().add_patch(
                plt.Rectangle((x1, y1 - 16), x2 - x1, 16, fill=True, edgecolor=ec, linewidth=1, facecolor=ec))
            plt.text(x1 + 4, y1 - 2, str(int(pid)), fontsize=12, color='w')
        plt.gca().add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=ec, linewidth=3))
    plt.show()