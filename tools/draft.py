import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import cPickle
import numpy as np
import caffe, os, sys, cv2
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

def im_detect_py_list(net, image_list):

    im_data = net.get_image_blob_list(image_list)
    # print im_data.shape
    cls_prob = net(im_data)
    scores = [cls_prob[i].data.cpu().numpy() for i in range(13)]
    return scores


def safe_log(x, minval=10e-40):
    return np.log(x.clip(min=minval))

def get_second(det):
    def get_possible_phone(arr):
            def bp(result, k):
                out = []
                for label in result:
                    for i in k:
                        out.append(label+[i])
                return out
            result = [[]]
            for k in arr:
                result = bp(result, k)
            return result

    def get_labels_rescaling_2(det):
        label = []
        score = []
        for arr in det:
            if arr[0] > arr[1]:
                large_ind, small_ind = 0, 1
                large, small = arr[0], arr[1]
            else:
                large_ind, small_ind = 1, 0
                large, small = arr[1], arr[0]
            for j in range(2, arr.shape[0]):
                if arr[j] > large:
                    small = large
                    large = arr[j]
                    small_ind = large_ind
                    large_ind = j
                elif arr[j] > small:
                    small = arr[j]
                    small_ind = j
                else:
                    continue
            label.append([large_ind, small_ind])
            score.append([large, small])
        return label, score
    det_probs_2 = get_labels_rescaling_2(det)[1]
    det_labels_2 = get_labels_rescaling_2(det)[0]
    det_labels_2_rectify = [[det_labels_2[i][0]] if det_probs_2[i][0] > np.log(0.98) else det_labels_2[i] for i in range(len(det_probs_2))]
    det_probs_2_rectify = [[det_probs_2[i][0]] if det_probs_2[i][0] > np.log(0.98) else det_probs_2[i] for i in range(len(det_probs_2))]

    res = get_possible_phone(det_labels_2_rectify)
    prob = get_possible_phone(det_probs_2_rectify)
    return res, prob



if __name__ == '__main__':
    det_thresh = 0.1
    phone_each_express = 3
    SECOND = False
    DET_IS_BG = -0.1

    test_def = os.path.join(os.getcwd(), 'models', 'express', 'VGG16', 'test.prototxt')
    caffemodel = os.path.join(os.getcwd(), 'output', 'express_train', 'express_iter_25000.caffemodel')
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net_rcnn = caffe.Net(test_def, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    net = PhoneNet()
    trained_model = os.path.join(os.getcwd(), 'output', 'phone_out', 'old', 'phone_25000.h5')
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    print '\n\nLoaded network {:s}'.format(trained_model)

    dataset_path = os.path.join(os.getcwd(), 'demo', 'dataset')
    phone_path = os.path.join(os.getcwd(), 'demo', 'crop_phone')
    if not os.path.isdir(dataset_path):
        print 'error data dir path {}'.format(dataset_path)
    if not os.path.isdir(phone_path):
        os.mkdir(phone_path)

    im_name = sorted(os.listdir(dataset_path))[60]
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
    cls_dets = cls_dets[:phone_each_express]
    print cls_dets

    score_all = []
    for i in range(cls_dets.shape[0]):
        x1, y1, x2, y2 = cls_dets[i][:4]
        if x2 > width / 2:
            continue

        # x1 = max(x1 - int(5 * width / 300.0), 0)
        x2 = max(x2 + int(5 * width / 300.0), 0)
        y1 = max(y1 - 3, 0)
        y2 = max(y2 + 3, 0)
        cropped_list = []
        for shift_grid in range(-5, 6):
            shift = int(shift_grid * width / 300.0)
            cropped = im[y1:y2, max(x1+shift, 0):max(x2+shift, 0), :]
            cropped_list.append(cropped)

        scores = im_detect_py_list(net, cropped_list)
        res_list = []
        sco_list = []
        for i in range(scores[0].shape[0]):
            phone_length = np.argmax(scores[-1][i]) + 5
            score = np.vstack([scores[j][i] for j in range(phone_length)])
            score = safe_log(score[:, :-1]).astype(np.float32)
            res = np.argmax(score, axis=1)
            res = ''.join(str(e) for e in list(res))
            sco = np.max(score, axis=1)
            sco_ave = np.mean(sco)
            res_list.append(res)
            sco_list.append(sco_ave)
        print res_list
        print sco_list



            
            