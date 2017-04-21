import os
import numpy as np
from scipy.sparse import csr_matrix
import datasets
from datasets.imdb import imdb
from fast_rcnn.config import cfg
import cPickle

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

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
        return os.path.join(cfg.DATA_DIR, 'express', 'pretrain_db')

    def _get_default_path_benchmark(self):
        return os.path.join(cfg.DATA_DIR, 'express', 'pretrain_db_benchmark')

    def evaluate_detections(self, all_boxes, output_dir):

        gt_roidb = self.gt_roidb()
        assert len(all_boxes) == len(gt_roidb)
        assert len(all_boxes[1]) == 13

        tp, num, tlength, phone_right = 0, 0, 0, 0
        extra_tp, extra_num = 0, 0
        certain_num = 0
        res_all = []
        phone_right_list = []
        mat = np.zeros((10, 11), dtype=np.int32)
        test_image_num = 0
        for gt, det in zip(gt_roidb, all_boxes):
            gt_labels = gt['labels']
            det_probs = [np.log(np.max(score)) for score in det]
            det_labels = [np.argmax(score) for score in det]
            det[-1][0] = np.where(det[-1][0] <= 10 ** -44, 10 ** -44, det[-1][0])
            det_length = np.log(det[-1][0])
            phone_score = []
            prob_sum = sum([det_probs[i] for i in range(4)])
            for i in range(8):
                prob_sum += det_probs[i+4]
                phone_score.append(prob_sum + det_length[i])
            # prob_sum = 0
            # for i in range(12):
            #     prob_sum += det_probs[i]
            #     phone_score.append(prob_sum + det_length[i])
            phone_length = np.argmax(phone_score) + 5

            def big_2(arr):
                large_ind, small_ind, large, small = 0, 0, 0, 0
                if arr[0] > arr[1]:
                    large_ind, small_ind = 0, 1
                    large, small = arr[0], arr[1]
                else:
                    large_ind, small_ind = 1, 0
                    large, small = arr[1], arr[0]
                for i in range(2, arr.shape[0]):
                    if arr[i] > large:
                        small = large
                        large = arr[i]
                        small_ind = large_ind
                        large_ind = i
                    elif arr[i] <= large and arr[i] >= small:
                        small = arr[i]
                        small_ind = i
                    else:
                        continue
                return [large, small], [large_ind, small_ind]
            
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

            def if_phone_right(list_a, arr_b):
                for a in list_a:
                    if (np.array(a) == arr_b).all():
                        return a
                return False

            if cfg.TEST.CERTAIN:
                det_probs_2 = [big_2(score[0])[0] for score in det][:phone_length]
                det_labels_2 = [big_2(score[0])[1] for score in det][:phone_length]
                det_labels_2_rectify = [[det_labels_2[i][0]] if det_probs_2[i][0] > 0.98 else det_labels_2[i] for i in range(len(det_probs_2))]

                res = get_possible_phone(det_labels_2_rectify)
                res = [labels[:phone_length] for labels in res]
                res_all.append(res)

                if len(res[0]) == gt_labels.shape[0]:
                    extra_num += 1
                    if if_phone_right(res, gt_labels):
                        extra_tp += 1
                if len(res[0]) > gt_labels.shape[0]:
                    res = [phone[:gt_labels.shape[0]] for phone in res]
                elif len(res[0]) < gt_labels.shape[0]:
                    res = [phone + ([10] * (gt_labels.shape[0] - len(phone))) for phone in res]
                else:
                    tlength += 1
                
                res1 = np.array(res[0])
                num += gt_labels.shape[0]
                tp += np.sum((res1 == gt_labels))
                for i, j in zip(gt_labels, res1):
                    mat[i][j] += 1
                if if_phone_right(res, gt_labels):
                    phone_right += 1
                    phone_right_list.append(test_image_num)
            else:
                res = det_labels[:phone_length]
                res_all.append(res)
                # test if len(res) == gt_labels.shape[0]
                if len(res) == gt_labels.shape[0]:
                    extra_num += 1
                    if (np.array(res) == gt_labels).all():
                        extra_tp += 1

                if len(res) > gt_labels.shape[0]:
                    res = res[:gt_labels.shape[0]]
                elif len(res) < gt_labels.shape[0]:
                    res.extend([10] * (gt_labels.shape[0] - len(res)))
                else:
                    tlength += 1
                res = np.array(res)
                num += gt_labels.shape[0]
                tp += np.sum((res == gt_labels))
                for i, j in zip(gt_labels, res):
                    mat[i][j] += 1
                if (res == gt_labels).all():
                    phone_right += 1
                    phone_right_list.append(test_image_num)
            test_image_num += 1

        cache_file = os.path.join(output_dir, 'digit_mat.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(mat, fid, cPickle.HIGHEST_PROTOCOL)
        cache_file = os.path.join(output_dir, 'detection_phone.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(res_all, fid, cPickle.HIGHEST_PROTOCOL)
        cache_file = os.path.join(output_dir, 'phone_right_list.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(phone_right_list, fid, cPickle.HIGHEST_PROTOCOL)
        if cfg.TEST.CERTAIN:
            print 'certain num:{:.4f}'.format(certain_num)
        print 'right digits: {:.4f}'.format(float(tp) / float(num))
        print 'right length: {:.4f}'.format(float(tlength) / float(len(gt_roidb)))
        print 'right phones: {:.4f}'.format(float(phone_right) / float(len(gt_roidb)))
        print 'extra right phones: {:.4f}'.format(float(extra_tp) / float(extra_num))
