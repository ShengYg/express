import os
import numpy as np
from scipy.sparse import csr_matrix
import datasets
from datasets.imdb import imdb
from fast_rcnn.config import cfg
import cPickle
import itertools
import heapq


def safe_log(x, minval=10e-40):
    return np.log(x.clip(min=minval))

class mnist(imdb):
    def __init__(self, image_set, root_dir=None, ratio=0.8):
        super(mnist, self).__init__('mnist_' + image_set)
        self._image_set = image_set
        self._training_ratio = ratio
        self._root_dir = self._get_default_path() if root_dir is None \
                         else root_dir
        self._data_path = os.path.join(self._root_dir, 'images')
        self._classes = ('zero', 'one', 'two', 'three',
                        'four', 'five', 'six', 'seven',
                        'eight', 'nine')
        self._namelist = self._load_namelist()
        self._info = self._load_info()
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        assert os.path.isdir(self._root_dir), \
                "mnist does not exist: {}".format(self._root_dir)
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
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print "{} gt roidb loaded from {}".format(self.name, cache_file)
        #     return roidb
        
        # Construct the gt_roidb
        gt_roidb = []
        for index in self.image_index:
            labels = self._info[index]
            gt_roidb.append({
                'labels': labels,
                'flipped' : False})
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print "wrote gt roidb to {}".format(cache_file)
        print "gt_roidb length: {}".format(len(gt_roidb))
        return gt_roidb

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'express', 'pretrain_mnist')

    def evaluate_detections(self, all_labels, output_dir, roidb):
        gt_roidb = roidb
        assert len(all_labels) == len(gt_roidb)

        tp = 0
        for gt, det in zip(gt_roidb, all_labels):
            gt_label = gt['labels']
            # det_label = np.argmax(det)
            det_label = det
            if gt_label == det_label:
                tp += 1
        print 'right digits: {} / {} = {:.4f}'.format(tp, len(gt_roidb), float(tp) / float(len(gt_roidb)))
