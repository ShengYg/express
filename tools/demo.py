import _init_paths
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from fast_rcnn.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
import pprint

import time
import cPickle
import numpy as np
import caffe, os, sys, cv2
from collections import Counter

import network
from phone_net import PhoneNet
from utils.timer import Timer

def im_detect_py(net, image):

    im_data = net.get_image_blob(image)
    cls_prob = net(im_data)
    scores = [cls_prob[i].data.cpu().numpy() for i in range(13)]
    return scores

def im_detect_py_list(net, image_list):

    im_data = net.get_image_blob_list(image_list)
    cls_prob = net(im_data)
    scores = [cls_prob[i].data.cpu().numpy() for i in range(13)]
    return scores

def safe_log(x, minval=10e-40):
    return np.log(x.clip(min=minval))

if __name__ == '__main__':

    cfg_from_file(os.path.join(os.getcwd(), 'experiments', 'cfgs', 'train.yml'))
    print('Using config:')
    pprint.pprint(cfg)

    RESIZE = True
    SHIFT = True

    det_thresh = 0.8
    phone_thres = -1
    NMS = 0.
    
    test_def = os.path.join(os.getcwd(), 'models', 'express', 'VGG16', 'test.prototxt')
    caffemodel = os.path.join(os.getcwd(), 'output', 'express_out', 'express_iter_25000_small.caffemodel')
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net_rcnn = caffe.Net(test_def, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    net = PhoneNet()
    trained_model = os.path.join(os.getcwd(), 'output', 'phone_out', 'phone_28000.h5')
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    print '\n\nLoaded network {:s}'.format(trained_model)

    dataset_path = os.path.join(os.getcwd(), 'demo', 'dataset')
    if not os.path.isdir(dataset_path):
        print 'error data dir path {}'.format(dataset_path)
    filelist = sorted(os.listdir(dataset_path))
    if not filelist:
        print 'no pic in dir {}'.format(dataset_path)
    print '\n\nFinded image path {:s}'.format(dataset_path)
    print '\n\n'

    img_num = 0
    for im_name in filelist:
        name = im_name.split('_')[0]
        print '{:.0f}%    {}'.format(float(img_num)/len(filelist)*100, im_name)
        img_num += 1
        im = cv2.imread(dataset_path + '/' + im_name)
        width = im.shape[1]

        scores, boxes = im_detect(net_rcnn, im)

        inds = np.where(scores[:, 1] > det_thresh)[0]
        cls_scores = scores[inds, 1]
        cls_boxes = boxes[inds, 4:]
        pos_ind = np.where(cls_boxes[:, 0] < width*0.4)[0]
        cls_scores = cls_scores[pos_ind]
        cls_boxes = cls_boxes[pos_ind]
        cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(cls_dets, NMS)
        cls_dets = cls_dets[keep, :]
        cls_dets = cls_dets[:3]

        for k in range(cls_dets.shape[0]):
            x1, y1, x2, y2 = cls_dets[k][:4].astype(np.int16)
            if RESIZE:
                x1 = max(x1 - int(5 * width / 300.0), 0)
                x2 = max(x2 + int(5 * width / 300.0), 0)
                y1 = max(y1 - 3, 0)
                y2 = max(y2 + 3, 0)

            if SHIFT:
                cropped_list = []
                for shift_grid in range(-10, 11):
                    shift = int(shift_grid * width / 600.0)
                    cropped = im[y1:y2, max(x1+shift, 0):max(x2+shift, 0), :]
                    cropped_list.append(cropped)
      
                scores = im_detect_py_list(net, cropped_list)
            else:
                cropped = im[y1:y2, x1:x2, :]
                scores = im_detect_py(net, cropped)

            res_list = []
            sco_list = []
            for i in range(scores[0].shape[0]):
                phone_length = np.argmax(scores[-1][i]) + 5
                score = np.vstack([scores[j][i] for j in range(phone_length)])
                score = safe_log(score[:, :-1]).astype(np.float32)
                res = np.argmax(score, axis=1)
                sco = np.max(score, axis=1)
                res_list.append(res)
                sco_list.append(np.mean(sco))

            ## use score_max
            ind = np.argmax(sco_list)
            res_after_select, score_after_select = res_list[ind], sco_list[ind]
            if score_after_select < phone_thres:
                continue
            print cls_dets[k][:4]
            print res_after_select