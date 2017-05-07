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
from progressbar import ProgressBar

import network
from phone_net import PhoneNet
from utils.timer import Timer

def im_detect_py(net, image):

    im_data = net.get_image_blob(image)
    # print im_data.shape
    cls_prob = net(im_data)
    scores = [cls_prob[i].data.cpu().numpy() for i in range(13)]
    return scores

def safe_log(x, minval=10e-40):
    return np.log(x.clip(min=minval))

def get_labels_rescaling(det):
    label = list(np.argmax(det, axis=1))
    score = list(np.argmax(det, axis=1))
    return label, score

if __name__ == '__main__':
    det_thresh = 0.1

    test_def = os.path.join(os.getcwd(), 'models', 'express', 'VGG16', 'test.prototxt')
    caffemodel = os.path.join(os.getcwd(), 'output', 'express_train', 'express_iter_25000.caffemodel')
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net_rcnn = caffe.Net(test_def, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    net = PhoneNet()
    trained_model = os.path.join(os.getcwd(), 'output', 'phone_out', '11', 'phone_25000.h5')
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    print '\n\nLoaded network {:s}'.format(trained_model)

    dataset_path = os.path.join(os.getcwd(), 'demo', 'dataset')
    phone_path = os.path.join(os.getcwd(), 'demo', 'crop_phone')
    if not os.path.isdir(dataset_path):
        print 'error data dir path {}'.format(dataset_path)
    filelist = sorted(os.listdir(dataset_path))
    if not filelist:
        print 'no pic in dir {}'.format(dataset_path)

    img_num = 0
    # pbar = ProgressBar(maxval=len(filelist))
    # pbar.start()
    # pbar_index = 0
    for im_name in filelist:
        print im_name
        im = cv2.imread(dataset_path + '/' + im_name)
        width = im.shape[1]
        scores, boxes = im_detect(net_rcnn, im)

        inds = np.where(scores[:, 1] > det_thresh)[0]
        cls_scores = scores[inds, 1]
        cls_boxes = boxes[inds, 4:]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(cls_dets, cfg.TEST.NMS)
        cls_dets = cls_dets[keep, :]
        cls_dets = cls_dets.astype(np.int16)

        for i in range(cls_dets.shape[0]):
            x1, y1, x2, y2 = cls_dets[i][:4]
            cropped = im[y1:y2, x1:x2, :]
            if x2 > width / 2:
                continue
            scores = im_detect_py(net, cropped)

            phone_length = np.argmax(scores[-1]) + 5
            scores = np.vstack((scores[:phone_length]))
            scores = safe_log(scores[:, :-1])
            res = [get_labels_rescaling(scores)[0]]
            print res[0]