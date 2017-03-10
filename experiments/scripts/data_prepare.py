import xml.etree.ElementTree as ET
import os
import numpy as np
import cPickle
import random
import cv2
from progressbar import ProgressBar
import time

def get_box(filename):
    if not os.path.exists(filename):
        print 'name error'
    tree = ET.ElementTree(file = filename)
    box_all = np.array([0,0,0,0])
    for elem in tree.iter():
        if elem.tag == 'markinfo':
            info = elem.attrib
            box_all = np.vstack((box_all, np.array([int(info['x']), int(info['y']), int(info['w']), int(info['h'])])))
    return box_all[1:] if len(box_all.shape) != 1 else np.array([[0,0,0,0]])

def get_phone_box(box):
    '''
    return: phone_num, phone_box
    '''
    def outbox(box): # (x1, y1, x2, y2)
        x = min(box[:, 0])
        y = min(box[:, 1])
        w = max(box[:, 2])-min(box[:, 0])
        h = max(box[:, 3])-min(box[:, 1])
        return np.array([[x, y, w, h]])
    
    ave_width = sum(box[:, 2]) / box.shape[0]
    box[:, 2] += box[:, 0]
    box[:, 3] += box[:, 1]

    if box.shape[0] > 12:  # horizontal cut
        x_dis = abs(np.hstack((box[1:, 0], box[-1,0])) - box[:, 0])
        ind = np.where(x_dis > ave_width * 2)[0] + 1
        boxes = np.split(box, ind)
        return (len(boxes), np.vstack(map(outbox, boxes)))
    else:
        return (1, np.vstack((outbox(box))))

def split_box_vertical(box):
    '''
    return: phone_num, phone_box
    '''
    if max(box[:, 1]) - min(box[:, 1]) < box_two_thres:
        phones, box_phone = get_phone_box(box)
        return phones, box_phone
    else:
        y_dis = abs(box[:, 1] - box[0, 1])
        inds1 = np.where(y_dis < box_height_thres)[0]
        inds2 = np.where(y_dis >= box_height_thres)[0]
        phones1, box_phone1 = split_box_vertical(box[inds1])
        phones2, box_phone2 = split_box_vertical(box[inds2])
        return phones1 + phones2, np.vstack((box_phone1, box_phone2))

if __name__ == '__main__':
    path = '/home/sy/code/re_id/data/expressdata/'
    box_two_thres = 60
    box_height_thres = 30
    info_all = []
    if not os.path.isdir(path):
        print 'error data dir path {}'.format(path)
    filelist = sorted(os.listdir(path))
    if not filelist:
        print 'no pic in dir {}'.format(path)
        
    pbar = ProgressBar(maxval=len(filelist))
    pbar.start()
    i = 0
    print 'process start'
    for filename in filelist:
        num, suffix = filename.split('.')[0], filename.split('.')[1]
        if suffix == 'xml':
            box_num = get_box(path + filename)
            info = [num + '.jpg', box_num]  #(filename, box_num, nums, box_phone, phones, reserved)
            if box_num.shape[0] < 4:
                info.extend([0, np.array([[0,0,0,0]]), 0, False])
                continue
            phones, box_phone = split_box_vertical(box_num)
            info.extend([box_num.shape[0], box_phone, phones, True])
            info_all.append(info)
        i += 1
        pbar.update(i)
    pbar.finish()

    cache_file = '/home/sy/code/re_id/data/info.pkl'
    with open(cache_file, 'wb') as fid:
        cPickle.dump(info_all, fid, cPickle.HIGHEST_PROTOCOL)