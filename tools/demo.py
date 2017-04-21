import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse



def vis_detections(im, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:.3f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    plt.axis('off')
    plt.tight_layout()
    plt.draw()


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True
    cfg.TRAIN.MAX_SIZE = 1300
    cfg.TRAIN.SCALES = (700, )

    prototxt = '/home/sy/code/re_id/express/models/express/VGG16/test.prototxt'
    caffemodel = '/home/sy/code/re_id/express/output/out/express_iter_25000.caffemodel'
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net_rcnn = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    prototxt = '/home/sy/code/re_id/express/models/phone/VGG16/test.prototxt'
    caffemodel = '/home/sy/code/re_id/express/output/out/express_iter_25000.caffemodel'
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net_phone = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    path = '/home/sy/code/re_id/express/demo'
    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']

    for im_name in im_names:
        im = cv2.imread(path + im_name)
        scores, boxes = im_detect(net_rcnn, im)

        inds = np.where(scores[:, 1] > thresh)[0]
        cls_scores = scores[inds, 1]
        cls_boxes = boxes[inds, 4:]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        if vis:
            vis_detections(im, cls_dets)
            plt.show()

        # cls_dets phone det

    