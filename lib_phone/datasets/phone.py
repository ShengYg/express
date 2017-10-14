import os
import numpy as np
from scipy.sparse import csr_matrix
import datasets
from datasets.imdb import imdb
from fast_rcnn.config import cfg
import cPickle
import itertools
import heapq

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def safe_log(x, minval=10e-40):
    return np.log(x.clip(min=minval))

class phone(imdb):
    def __init__(self, image_set, root_dir=None, ratio=0.8):
        super(phone, self).__init__('phone_' + image_set)
        self._image_set = image_set
        self._training_ratio = ratio
        self._root_dir = self._get_default_path_benchmark() if root_dir is None \
                         else root_dir
        self._data_path = os.path.join(self._root_dir, 'images')
        self._classes = ('zero', 'one', 'two', 'three',
                        'four', 'five', 'six', 'seven',
                        'eight', 'nine', '__background__')
        self._namelist = self._load_namelist()
        self._info = self._load_info()
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        assert os.path.isdir(self._root_dir), \
                "phone does not exist: {}".format(self._root_dir)
        assert os.path.isdir(self._data_path), \
                "Path does not exist: {}".format(self._data_path)
        self.config = {'rpn_file'    : None}

    def _load_namelist(self):
        namelist_path = os.path.join(self._root_dir, 'namelist.pkl')
        if os.path.exists(namelist_path):
            with open(namelist_path, 'rb') as fid:
                return cPickle.load(fid)
        else:
            raise Exception('No namelist.pkl, init error')

    def _load_info(self):
        info_path = os.path.join(self._root_dir, 'info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as fid:
                return cPickle.load(fid)
        else:
            raise Exception('No info.pkl, init error')

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, index)
        assert os.path.isfile(image_path), \
                "Path does not exist: {}".format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes for the specific subset (train / test).
        For express, the index is just the image file name.
        """
        train_num = int(len(self._namelist) * self._training_ratio)
        if self._image_set == 'train':
            return self._namelist[:train_num]
        elif self._image_set == 'test':
            return self._namelist[train_num:]

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        gt_roidb = []
        for index in self.image_index:
            labels = self._info[index][0] + 1
            bbox = self._info[index][1]             # x, y, w, h
            im_size = self._info[index][2]
            boxes = np.copy(self._info[index][3])   # x1, y1, x2, y2

            num_objs = boxes.shape[0]
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            for i in range(num_objs):
                overlaps[:, labels[i]] = 1.0
            overlaps = csr_matrix(overlaps)

            gt_roidb.append({
                'image': index,
                'bbox': bbox,
                'boxes': boxes,
                'gt_classes': labels,
                'gt_overlaps': overlaps,
                'flipped': False,
                'im_size': im_size})
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print "wrote gt roidb to {}".format(cache_file)
        print "gt_roidb length: {}".format(len(gt_roidb))
        return gt_roidb

    def _get_default_path_benchmark(self):
        return os.path.join(cfg.DATA_DIR, 'express', 'pretrain_db_benchmark')

    def evaluate_detections(self, all_boxes, output_dir):
        # all_boxes[cls][image] = N x 5 (x1, y1, x2, y2, score)
        gt_roidb = self.gt_roidb()
        assert len(all_boxes) == len(gt_roidb)
        assert len(all_boxes[0]) == 11

        all_length = [0] * 8
        each_length = [0] * 8
        phone_right_each = [0] * 8
        tp, num, tlength, phone_right = 0, 0, 0, 0
        tp_arr = np.zeros((12, ))
        all_arr = np.zeros((12, ))
        extra_tp, extra_num = 0, 0
        res_all = []
        phone_wrong_list = []
        mat = np.zeros((10, 11), dtype=np.int32)
        test_image_num = 0
        for gt, det in zip(gt_roidb, all_boxes):
            gt_boxes = gt['boxes']
            gt_labels = gt['gt_classes']
            cls_det = np.vstack([np.hstack((det[i], np.array([i]*det[i].shape[0]).reshape(-1, 1))) for i in range(1,11)])
            cls_det = cls_det[np.argsort(np.mean(cls_det[:, [0,2]], axis=1))]
            res = cls_det[:, 5]

            all_length[gt_labels.shape[0]-5] += 1
            if res.shape[0] > gt_labels.shape[0]:
                res = res[:gt_labels.shape[0]]
            elif res.shape[0] < gt_labels.shape[0]:
                res = np.hstack([res, np.array([10]*(gt_labels.shape[0]-res.shape[0]))])
            else:
                tlength += 1
                each_length[gt_labels.shape[0]-5] += 1
            
            num += gt_labels.shape[0]
            tp += np.sum((res == gt_labels))
            tp_arr[np.where(res == gt_labels)] += 1
            all_arr[np.arange(gt_labels.shape[0])] += 1
            if np.all(res == gt_labels):
                phone_right += 1
                phone_right_each[gt_labels.shape[0]-5] += 1
            else:
                phone_wrong_list.append(test_image_num)
            test_image_num += 1

        cache_file = os.path.join(output_dir, 'detection_phone.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(res_all, fid, cPickle.HIGHEST_PROTOCOL)
        cache_file = os.path.join(output_dir, 'phone_wrong_list.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(phone_wrong_list, fid, cPickle.HIGHEST_PROTOCOL)
        print 'right digits: {} / {} = {:.4f}'.format(tp, num, float(tp) / float(num))
        print 'right length: {} / {} = {:.4f}'.format(tlength, len(gt_roidb), float(tlength) / float(len(gt_roidb)))
        print 'right phones: {} / {} = {:.4f}'.format(phone_right, len(gt_roidb), float(phone_right) / float(len(gt_roidb)))

        print 'all: {}'.format(all_length)
        print 'length right: {}'.format(each_length)
        print 'phone right: {}'.format(phone_right_each)

        print 'the following is acc in each pos from 1 to 12'
        print 'right: {}'.format(tp_arr)
        print 'all  : {}'.format(all_arr)
        print 'acc  : {}'.format(tp_arr.astype(np.float32) / all_arr)

            

