import _init_paths
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms

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

if __name__ == '__main__':

    det_thresh = 0.5
    NMS = 0.
    down_line_thres = 100
    down_line_dis_thres = 30

    test_def = os.path.join(os.getcwd(), 'models', 'express', 'VGG16', 'test.prototxt')
    caffemodel = os.path.join(os.getcwd(), 'output', 'express_train', 'express_iter_25000.caffemodel')
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net_rcnn = caffe.Net(test_def, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    net = PhoneNet()
    trained_model = os.path.join(os.getcwd(), 'output', 'phone_train', 'phone_25000.h5')
    # trained_model = os.path.join(os.getcwd(), 'output', 'phone_out', '11', 'phone_25000.h5')
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

    filelist = sorted(os.listdir(dataset_path))
    if not filelist:
        print 'no pic in dir {}'.format(dataset_path)

    all_boxes = {}
    cache_file = os.path.join(os.getcwd(), 'demo', 'img_detections.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            all_boxes = cPickle.load(fid)
    score_all = {}
    img_num = 0
    time_all, cnt = 0, 0
    for im_name in filelist:
        name = im_name.split('_')[0]
        cls_dets = None
        print '{:.0f}%    {}'.format(float(img_num)/len(filelist)*100, im_name)
        im = cv2.imread(dataset_path + '/' + im_name)
        width = im.shape[1]
        if name not in all_boxes:
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

            name = im_name.split('_')[0]
            all_boxes[name] = cls_dets
        else:
            cls_dets = all_boxes[name]

        for i in range(cls_dets.shape[0]):
            x1, y1, x2, y2 = cls_dets[i][:4].astype(np.int16)
            ## add resize
            # x1 = max(x1 - int(5 * width / 300.0), 0)
            x2 = max(x2 + int(5 * width / 300.0), 0)
            y1 = max(y1 - 3, 0)
            y2 = max(y2 + 3, 0)
            crop_img_name = name + '_{}.jpg'.format(i)

            start_time = time.time()
            cropped_list = []
            for shift_grid in range(-5, 6):
                shift = int(shift_grid * width / 300.0)
                cropped = im[y1:y2, max(x1+shift, 0):max(x2+shift, 0), :]
                # cv2.imwrite(os.path.join(phone_path, crop_img_name), cropped)
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
                res_list.append(res)
                sco_list.append(sco)
            ## use count
            # d = Counter(res_list)
            # max_cnt, max_value = 0, 0
            # for item in d:
            #     if d[item] > max_cnt:
            #         max_cnt = d[item]
            #         max_value = item
            # sco_out = []
            # for r, s in zip(res_list, sco_list):
            #     if r == max_value:
            #         sco_out.append(s)

            # time_all += time.time() - start_time
            # cnt += 1
            # score_all[crop_img_name] = [np.array([int(s) for s in list(max_value)]), np.mean(np.vstack(sco_out))]

            ## use score_max
            sco_list = [np.mean(item) for item in sco_list]
            ind = np.argmax(sco_list)
            time_all += time.time() - start_time
            cnt += 1
            score_all[crop_img_name] = [np.array([int(s) for s in list(res_list[ind])]), sco_list[ind]]

        img_num += 1

    print 'ave_time: {}'.format(float(time_all) / cnt)
    cache_file = os.path.join(os.getcwd(), 'demo', 'img_namelist.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(filelist, fid, cPickle.HIGHEST_PROTOCOL)
    cache_file = os.path.join(os.getcwd(), 'demo', 'img_detections.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(all_boxes, fid, cPickle.HIGHEST_PROTOCOL)
    cache_file = os.path.join(os.getcwd(), 'demo', 'phone_score.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(score_all, fid, cPickle.HIGHEST_PROTOCOL)
