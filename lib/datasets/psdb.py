import os
import numpy as np
from scipy.sparse import csr_matrix
from scipy.io import loadmat

import datasets
from datasets.imdb import imdb
from fast_rcnn.config import cfg
from utils import cython_bbox
import cPickle
import random
import cv2

'''
roidb :(n,)list[{}, {}, ...]
boxes           (num_boxes, 4)
gt_classes      (num_boxes, ) = 1
gt_overlaps     (num_boxes, num_classes)
gt_pids         (num_boxes, )
flipped         False
'''
# how many images does one person appear: [0, 0, 3056, 1534, 568, 251, 74, 28, 9, 3, 1, 4, 1, 1, 1, 1]
# sum: 5532

#how many person appears in a image:

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union


class psdbcrop(imdb):
    def __init__(self, image_set, root_dir=None):
        super(psdbcrop, self).__init__('psdbcrop_' + image_set)
        self._image_set = image_set
        self._root_dir = self._get_default_path() if root_dir is None \
                         else root_dir
        self._data_path = os.path.join(self._root_dir, 'Image', 'SSM')
        self._classes = ('__background__', 'person')
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self._pid_and_name = []
        self._pid_and_index = []
        self._input_list = self.input_list
        assert os.path.isdir(self._root_dir), \
                "PSDB does not exist: {}".format(self._root_dir)
        assert os.path.isdir(self._data_path), \
                "Path does not exist: {}".format(self._data_path)
        self.config = {'rpn_file'    : None}

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, index)
        assert os.path.isfile(image_path), \
                "Path does not exist: {}".format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print "{} gt roidb loaded from {}".format(self.name, cache_file)
            return roidb

        cache_file1 = os.path.join(self.cache_path, self.name + '_box_and_pid.pkl')
        if os.path.exists(cache_file1):
            print "====================================================================="
            print "second time, loading boxes and pids"
            with open(cache_file1, 'rb') as fid:
                name_to_boxes, name_to_pids = cPickle.load(fid)
            print "{} boxes and pids loaded from {}".format(self.name, cache_file1)
        else:
            print "====================================================================="
            print "first time, saving resized images"

            def crop(boxes, im_name):
                w, h = 500, 300
                wide, narrow = 50, 10
                boxes[:, 2] += boxes[:, 0]
                boxes[:, 3] += boxes[:, 1]
                # if im_name == 's15045.jpg':
                #     print boxes
                width, height = max(boxes[:, 2]) - min(boxes[:, 0]), max(boxes[:, 3]) - min(boxes[:, 1])
                x0, y0 = min(boxes[:, 0]), min(boxes[:, 1])
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]
                
                path = '/home/sy/code/re_id/data/psdb/dataset/Image/SSM/'
                path_new = '/home/sy/code/re_id/data/psdb_new/dataset/Image/SSM1/'
                im = cv2.imread(path + im_name)
                if width <= w and height <= h:
                    im_new = im[max(y0-wide, 0):min(y0+height+wide, im.shape[0] - 1), max(x0-wide, 0):min(x0+width+wide, im.shape[1] - 1), :]
                    boxes[:, 0] -= max(x0-wide, 0)
                    boxes[:, 1] -= max(y0-wide, 0)
                    cv2.imwrite(path_new + im_name, im_new)
                elif width <= w:
                    im_new = im[max(y0-narrow, 0):min(y0+height+narrow, im.shape[0] - 1), max(x0-wide, 0):min(x0+width+wide, im.shape[1] - 1), :]
                    boxes[:, 0] -= max(x0-wide, 0)
                    boxes[:, 1] -= max(y0-narrow, 0)
                    cv2.imwrite(path_new + im_name, im_new)
                elif height <= h:
                    im_new = im[max(y0-wide, 0):min(y0+height+wide, im.shape[0] - 1), max(x0-narrow, 0):min(x0+width+narrow, im.shape[1] - 1), :]
                    boxes[:, 0] -= max(x0-narrow, 0)
                    boxes[:, 1] -= max(y0-wide, 0)
                    cv2.imwrite(path_new + im_name, im_new)
                else:
                    im_new = im[max(y0-narrow, 0):min(y0+height+narrow, im.shape[0] - 1), max(x0-narrow, 0):min(x0+width+narrow, im.shape[1] - 1), :]
                    boxes[:, 0] -= max(x0-narrow, 0)
                    boxes[:, 1] -= max(y0-narrow, 0)
                    cv2.imwrite(path_new + im_name, im_new)
                return boxes

            # Load all images and build a dict from image to boxes
            all_imgs = loadmat(os.path.join(self._root_dir, 'annotation', 'Images.mat'))
            all_imgs = all_imgs['Img'].squeeze()
            name_to_boxes = {}
            name_to_pids = {}
            for im_name, __, boxes in all_imgs:
                im_name = str(im_name[0])       # u's14860.jpg'
                boxes = np.asarray([b[0] for b in boxes[0]]).astype(np.int32)
                boxes = boxes.reshape(boxes.shape[0], 4)        # n * 4
                valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
                assert valid_index.size > 0, \
                    'Warning: {} has no valid boxes.'.format(im_name)
                boxes = boxes[valid_index]
                if im_name in self.image_index:
                    old_boxes = np.copy(boxes)
                    boxes = crop(boxes, im_name)
                    name_to_boxes[im_name] = (boxes.astype(np.int32), old_boxes.astype(np.int32))
                else:
                    name_to_boxes[im_name] = (boxes.astype(np.int32), boxes.astype(np.int32))

                # name_to_boxes[im_name] = boxes.astype(np.int32)
                name_to_pids[im_name] = -1 * np.ones(boxes.shape[0], dtype=np.int32)
                #pid includes image query/gallery num

            def _set_box_pid(boxes, box, pids, pid):
                for i in xrange(boxes.shape[0]):
                    if np.all(boxes[i] == box):
                        pids[i] = pid
                        return
                print 'Warning: person {} box {} cannot find in Images'.format(pid, box)
            def _get_box(boxes, box):
                for i in xrange(boxes[1].shape[0]):
                    if np.all(boxes[1][i] == box):
                        return boxes[0][i]
                return np.array([], dtype = np.int32).reshape(1, 0)

            if self._image_set == 'train':
                train = loadmat(os.path.join(self._root_dir,
                                         'annotation/test/train_test/Train.mat'))
                train = train['Train'].squeeze()
                for index, item in enumerate(train):
                    scenes = item[0, 0][2].squeeze()
                    # pid_name_list = []
                    for im_name, box, __ in scenes:
                        im_name = str(im_name[0])
                        box = box.squeeze().astype(np.int32)
                        _set_box_pid(name_to_boxes[im_name][1], box,
                                     name_to_pids[im_name], index)
            else:
                test = loadmat(os.path.join(self._root_dir,
                                        'annotation/test/train_test/TestG50.mat'))
                test = test['TestG50'].squeeze()
                dt = np.dtype([('imname', 'O'), ('idlocate', 'O')])
                test_crop = {}
                test_crop['Query'] = []
                test_crop['Gallery'] = []
                for index, item in enumerate(test):
                    # query
                    im_name = str(item['Query'][0,0][0][0])
                    box = item['Query'][0,0][1].squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name][1], box,
                                 name_to_pids[im_name], index)
                    test_crop['Query'].append(np.array((im_name, _get_box(name_to_boxes[im_name], box)), dtype = dt))
                    # gallery
                    gallery = item['Gallery'].squeeze()
                    gallery_crop = []
                    for im_name, box, __ in gallery:
                        im_name = str(im_name[0])
                        box = box.squeeze().astype(np.int32)
                        if box.size != 0:
                            _set_box_pid(name_to_boxes[im_name][1], box,
                                         name_to_pids[im_name], index)
                        gallery_crop.append(np.array((im_name, _get_box(name_to_boxes[im_name], box)), dtype = dt))
                    test_crop['Gallery'].append(gallery_crop)
                inportant_file = os.path.join(self.cache_path, self.name + '_test_crop.pkl')
                with open(inportant_file, 'wb') as fid:
                    cPickle.dump(test_crop, fid, cPickle.HIGHEST_PROTOCOL)
            cache_file1 = os.path.join(self.cache_path, self.name + '_box_and_pid.pkl')
            with open(cache_file1, 'wb') as fid:
                cPickle.dump((name_to_boxes, name_to_pids), fid, cPickle.HIGHEST_PROTOCOL)

        
        # Construct the gt_roidb
        gt_roidb = []
        for index in self.image_index:
            boxes = name_to_boxes[index][0]
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            pids = name_to_pids[index]
            num_objs = len(boxes)
            gt_classes = np.ones((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32) 
            # num_classes: maybe 0
            overlaps[:, 1] = 1.0
            overlaps = csr_matrix(overlaps)
            gt_roidb.append({
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'gt_pids': pids,
                'flipped': False})

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print "wrote gt roidb to {}".format(cache_file)
        return gt_roidb

    def input_list(self):
        train = loadmat(os.path.join(self._root_dir,
                                 'annotation/test/train_test/Train.mat'))
        train = train['Train'].squeeze()
        for index, item in enumerate(train):    #index: train query num
            scenes = item[0, 0][2].squeeze()
            pid_name_list = []
            for im_name, box, __ in scenes:
                im_name = str(im_name[0])
                pid_name_list.append(im_name)
        #         box = box.squeeze().astype(np.int32)
        #         _set_box_pid(name_to_boxes[im_name], box,
        #                      name_to_pids[im_name], index)
            self._pid_and_name.append(pid_name_list)
        input_list_cache = os.path.join(self.cache_path, self.name + '_name.pkl')
        with open(input_list_cache, 'wb') as fid:
            cPickle.dump(self._pid_and_name, fid, cPickle.HIGHEST_PROTOCOL)


        # _pid_and_index
        for item in self._pid_and_name:
            pid_index_list = []
            for i in item:
                index = self._image_index.index(i)
                pid_index_list.append(index)
                pid_index_list.append(index + self.num_images / 2)
            self._pid_and_index.append(pid_index_list)
        input_list_cache = os.path.join(self.cache_path, self.name + '_index.pkl')
        with open(input_list_cache, 'wb') as fid:
            cPickle.dump(self._pid_and_index, fid, cPickle.HIGHEST_PROTOCOL)

        return self._pid_and_index

    def evaluate_detections(self, all_boxes, output_dir):
        # all_boxes[cls][image] = N x 5 (x1, y1, x2, y2, score)
        gt_roidb = self.gt_roidb()
        assert len(all_boxes) == 2
        assert len(all_boxes[1]) == len(gt_roidb)
        y_true = []
        y_score = []
        count_gt = 0
        count_tp = 0
        for gt, det in zip(gt_roidb, all_boxes[1]):
            gt_boxes = gt['boxes']
            det = np.asarray(det)
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in xrange(num_gt):
                for j in xrange(num_det):
                    ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
            tfmat = (ious >= 0.5)
            # for each det, keep only the largest iou of all the gt
            for j in xrange(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in xrange(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False
            # for each gt, keep only the largest iou of all the det
            for i in xrange(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in xrange(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False

            for j in xrange(num_det):
                y_score.append(det[j, -1])
                if tfmat[:, j].any():
                    y_true.append(True)
                else:
                    y_true.append(False)
            count_tp += tfmat.sum()
            count_gt += num_gt

        from sklearn.metrics import average_precision_score, precision_recall_curve
        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate
        precision, recall, __ = precision_recall_curve(y_true, y_score)
        recall *= det_rate
        print 'mAP_person: {:.4%}'.format(ap)
        # import matplotlib.pyplot as plt
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.title('Det Rate: {:.2%}  AP: {:.2%}'.format(det_rate, ap))
        # plt.show()

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'psdb', 'dataset')

    def _load_image_set_index(self):
        """
        Load the indexes for the specific subset (train / test).
        For PSDB, the index is just the image file name.
        """
        # test pool
        test = loadmat(os.path.join(self._root_dir, 'annotation', 'pool.mat'))
        test = test['pool'].squeeze()
        test = [str(a[0]) for a in test]    # "[u's15535.jpg']"
        if self._image_set == 'test': 
            input_list_cache = os.path.join(self.cache_path, self.name + '_imageset_index.pkl')
            with open(input_list_cache, 'wb') as fid:
                cPickle.dump(test, fid, cPickle.HIGHEST_PROTOCOL)
            return test
        # all images
        all_imgs = loadmat(os.path.join(self._root_dir, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        all_list = list(set(all_imgs) - set(test))
        # training
        input_list_cache = os.path.join(self.cache_path, self.name + '_imageset_index.pkl')
        with open(input_list_cache, 'wb') as fid:
            cPickle.dump(all_list, fid, cPickle.HIGHEST_PROTOCOL)
        return all_list

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

class psdb(imdb):
    def __init__(self, image_set, root_dir=None):
        super(psdb, self).__init__('psdb_' + image_set)
        self._image_set = image_set
        self._root_dir = self._get_default_path() if root_dir is None \
                         else root_dir
        self._data_path = os.path.join(self._root_dir, 'Image', 'SSM')
        self._classes = ('__background__', 'person')
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self._pid_and_name = []
        self._pid_and_index = []
        self._input_list = self.input_list
        assert os.path.isdir(self._root_dir), \
                "PSDB does not exist: {}".format(self._root_dir)
        assert os.path.isdir(self._data_path), \
                "Path does not exist: {}".format(self._data_path)
        self.config = {'rpn_file'    : None}

    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        image_path = os.path.join(self._data_path, index)
        assert os.path.isfile(image_path), \
                "Path does not exist: {}".format(image_path)
        return image_path

    def gt_roidb(self):
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print "{} gt roidb loaded from {}".format(self.name, cache_file)
            return roidb

        # Load all images and build a dict from image to boxes
        all_imgs = loadmat(os.path.join(self._root_dir, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        for im_name, __, boxes in all_imgs:
            im_name = str(im_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, \
                'Warning: {} has no valid boxes.'.format(im_name)
            boxes = boxes[valid_index]
            name_to_boxes[im_name] = boxes.astype(np.int32)
            name_to_pids[im_name] = -1 * np.ones(boxes.shape[0], dtype=np.int32)

        def _set_box_pid(boxes, box, pids, pid):
            for i in xrange(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return
            print 'Warning: person {} box {} cannot find in Images'.format(pid, box)

        if self._image_set == 'train':
            train = loadmat(os.path.join(self._root_dir,
                                     'annotation/test/train_test/Train.mat'))
            train = train['Train'].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for im_name, box, __ in scenes:
                    im_name = str(im_name[0])
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box,
                                 name_to_pids[im_name], index)
        else:
            test = loadmat(os.path.join(self._root_dir,
                                    'annotation/test/train_test/TestG50.mat'))
            test = test['TestG50'].squeeze()
            for index, item in enumerate(test):
                # query
                im_name = str(item['Query'][0,0][0][0])
                box = item['Query'][0,0][1].squeeze().astype(np.int32)
                _set_box_pid(name_to_boxes[im_name], box,
                             name_to_pids[im_name], index)
                # gallery
                gallery = item['Gallery'].squeeze()
                for im_name, box, __ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0: break
                    box = box.squeeze().astype(np.int32)
                    _set_box_pid(name_to_boxes[im_name], box,
                                 name_to_pids[im_name], index)
        
        # Construct the gt_roidb
        gt_roidb = []
        for index in self.image_index:
            boxes = name_to_boxes[index]
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            pids = name_to_pids[index]
            num_objs = len(boxes)
            gt_classes = np.ones((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            overlaps[:, 1] = 1.0
            overlaps = csr_matrix(overlaps)
            gt_roidb.append({
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'gt_pids': pids,
                'flipped': False})
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print "wrote gt roidb to {}".format(cache_file)
        return gt_roidb

    def input_list(self):
        train = loadmat(os.path.join(self._root_dir,
                                 'annotation/test/train_test/Train.mat'))
        train = train['Train'].squeeze()
        for index, item in enumerate(train):    #index: train query num
            scenes = item[0, 0][2].squeeze()
            pid_name_list = []
            for im_name, box, __ in scenes:
                im_name = str(im_name[0])
                pid_name_list.append(im_name)
        #         box = box.squeeze().astype(np.int32)
        #         _set_box_pid(name_to_boxes[im_name], box,
        #                      name_to_pids[im_name], index)
            self._pid_and_name.append(pid_name_list)
        input_list_cache = os.path.join(self.cache_path, self.name + '_name.pkl')
        with open(input_list_cache, 'wb') as fid:
            cPickle.dump(self._pid_and_name, fid, cPickle.HIGHEST_PROTOCOL)


        # _pid_and_index
        for item in self._pid_and_name:
            pid_index_list = []
            for i in item:
                index = self._image_index.index(i)
                pid_index_list.append(index)
                pid_index_list.append(index + self.num_images / 2)
            self._pid_and_index.append(pid_index_list)
        input_list_cache = os.path.join(self.cache_path, self.name + '_index.pkl')
        with open(input_list_cache, 'wb') as fid:
            cPickle.dump(self._pid_and_index, fid, cPickle.HIGHEST_PROTOCOL)

        return self._pid_and_index

    def evaluate_detections(self, all_boxes, output_dir):
        # all_boxes[cls][image] = N x 5 (x1, y1, x2, y2, score)
        gt_roidb = self.gt_roidb()
        assert len(all_boxes) == 2
        assert len(all_boxes[1]) == len(gt_roidb)
        y_true = []
        y_score = []
        count_gt = 0
        count_tp = 0
        for gt, det in zip(gt_roidb, all_boxes[1]):
            gt_boxes = gt['boxes']
            det = np.asarray(det)
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in xrange(num_gt):
                for j in xrange(num_det):
                    ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
            tfmat = (ious >= 0.5)
            # for each det, keep only the largest iou of all the gt
            for j in xrange(num_det):
                largest_ind = np.argmax(ious[:, j])
                for i in xrange(num_gt):
                    if i != largest_ind:
                        tfmat[i, j] = False
            # for each gt, keep only the largest iou of all the det
            for i in xrange(num_gt):
                largest_ind = np.argmax(ious[i, :])
                for j in xrange(num_det):
                    if j != largest_ind:
                        tfmat[i, j] = False

            for j in xrange(num_det):
                y_score.append(det[j, -1])
                if tfmat[:, j].any():
                    y_true.append(True)
                else:
                    y_true.append(False)
            count_tp += tfmat.sum()
            count_gt += num_gt

        from sklearn.metrics import average_precision_score, precision_recall_curve
        det_rate = count_tp * 1.0 / count_gt
        ap = average_precision_score(y_true, y_score) * det_rate
        precision, recall, __ = precision_recall_curve(y_true, y_score)
        recall *= det_rate
        print 'mAP_person: {:.4%}'.format(ap)
        # import matplotlib.pyplot as plt
        # plt.plot(recall, precision)
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.title('Det Rate: {:.2%}  AP: {:.2%}'.format(det_rate, ap))
        # plt.show()

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'psdb', 'dataset')

    def _load_image_set_index(self):
        """
        Load the indexes for the specific subset (train / test).
        For PSDB, the index is just the image file name.
        """
        # test pool
        test = loadmat(os.path.join(self._root_dir, 'annotation', 'pool.mat'))
        test = test['pool'].squeeze()
        test = [str(a[0]) for a in test]    # "[u's15535.jpg']"
        if self._image_set == 'test': 
            input_list_cache = os.path.join(self.cache_path, self.name + '_imageset_index.pkl')
            with open(input_list_cache, 'wb') as fid:
                cPickle.dump(test, fid, cPickle.HIGHEST_PROTOCOL)
            return test
        # all images
        all_imgs = loadmat(os.path.join(self._root_dir, 'annotation', 'Images.mat'))
        all_imgs = all_imgs['Img'].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        all_list = list(set(all_imgs) - set(test))
        # training
        input_list_cache = os.path.join(self.cache_path, self.name + '_imageset_index.pkl')
        with open(input_list_cache, 'wb') as fid:
            cPickle.dump(all_list, fid, cPickle.HIGHEST_PROTOCOL)
        return all_list

    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

if __name__ == '__main__':
    from datasets.psdbcrop import psdbcrop
    d = psdbcrop('train')
    res = d.roidb
    from IPython import embed; embed()
