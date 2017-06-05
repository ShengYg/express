import _init_paths
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer

import cPickle
import numpy as np
import caffe, os, sys, cv2

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
    score_all = {}
    score_namelist = []
    img_num = 0
    for im_name in filelist:
        print '{:.0f}%    {}'.format(float(img_num)/len(filelist)*100, im_name)
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

        # for num in range(cls_dets.shape[0]):
        #     box = cls_dets[num][:4].astype(np.int16)
        #     img_i = im[box[1]:box[3], box[0]:box[2]]

        #     horizontal = np.sum(img_i[:,:,0] < 127, axis=1)
        #     h_line1 = []
        #     for i in range(1, horizontal.shape[0]):
        #         if horizontal[i] - horizontal[i-1] > down_line_thres:
        #             h_line1.append(i)
        #     if h_line1 and horizontal.shape[0] - h_line1[-1] < down_line_dis_thres:
        #         cls_dets[num][3] = cls_dets[num][1] + h_line1[-1]

        name = im_name.split('_')[0]
        all_boxes[name] = cls_dets
        # if cls_dets.shape[0] > 3:
        #     print '============================================='
        #     print im_name
        #     print cls_dets
        #     print '============================================='

        for i in range(cls_dets.shape[0]):
            x1, y1, x2, y2 = cls_dets[i][:4].astype(np.int16)
            cropped = im[y1:y2, x1:x2, :]
            crop_img_name = name + '_{}.jpg'.format(i)
            score_namelist.append(crop_img_name)
            cv2.imwrite(os.path.join(phone_path, crop_img_name), cropped)
            scores = im_detect_py(net, cropped)

            phone_length = np.argmax(scores[-1]) + 5
            scores = np.vstack((scores[:phone_length]))
            scores = safe_log(scores[:, :-1]).astype(np.float32)
            score_all[crop_img_name] = scores
        img_num += 1

    cache_file = os.path.join(os.getcwd(), 'demo', 'img_namelist.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(filelist, fid, cPickle.HIGHEST_PROTOCOL)
    cache_file = os.path.join(os.getcwd(), 'demo', 'img_detections.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(all_boxes, fid, cPickle.HIGHEST_PROTOCOL)
    cache_file = os.path.join(os.getcwd(), 'demo', 'score_namelist.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(score_namelist, fid, cPickle.HIGHEST_PROTOCOL)
    cache_file = os.path.join(os.getcwd(), 'demo', 'phone_score.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(score_all, fid, cPickle.HIGHEST_PROTOCOL)
