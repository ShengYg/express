import os
import numpy as np
from scipy.sparse import csr_matrix
import datasets
from datasets.imdb import imdb
from fast_rcnn.config import cfg
import cPickle
import heapq

def get_dis(s1, s2):
    # get distance between s1 and s2, s1=[1,2,3...]
    last = 0
    tmp = range(len(s2) + 1)
    value = None

    for i in range(len(s1)):
        tmp[0] = i + 1
        last = i
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                value = last
            else:
                value = 1 + min(last, tmp[j], tmp[j + 1])
            last = tmp[j+1]
            tmp[j+1] = value
    return value if value < 4 else 4

def LCS(str1, str2):
    n1, n2 = len(str1), len(str2)
    vec = [[0]*(n2+1) for _ in range(n1+1)]
    
    for i in range(1,n1+1):
        for j in range(1, n2+1):
            if str1[i-1]==str2[j-1]:
                vec[i][j]=vec[i-1][j-1]+1
            else:
                vec[i][j]=max(vec[i][j-1],vec[i-1][j])

    return vec[n1][n2]

def get_measure(pred, gt):
    # get precision and recall between pred and ft, pred=[1,2,3...]
    lcs_len = LCS(pred, gt)
    tp = float(lcs_len)
    fn = len(gt) - lcs_len
    fp = len(pred) - lcs_len
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    return precision, recall

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
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = cPickle.load(fid)
        #     print "{} gt roidb loaded from {}".format(self.name, cache_file)
        #     return roidb
        
        # Construct the gt_roidb
        gt_roidb = []
        for index in self.image_index:
            labels = self._info[index][0]
            bbox = self._info[index][1]
            gt_roidb.append({
                'labels': labels,
                'bbox': bbox,
                'image': index,
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

    def evaluate_detections(self, all_boxes, output_dir, roidb, weights=np.ones((12, 10)), length_weights=np.ones((8,))):
        def get_labels_rescaling(det):
            label = list(np.argmax(det, axis=1))
            score = list(np.max(det, axis=1))
            return label, score

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
                    return True
            return False

        def merge(a, b, ind_b, index, k):
            heap = []
            index_new = [None for i in range(k)]
            for i in range(k):
                heapq.heappush(heap, [-(a[i] + b[0]), i, 0])
            for i in range(k):
                item = heapq.heappop(heap)
                index_new[i] = index[item[1]] + [ind_b[item[2]]]
                a[i] = -item[0]
                j = item[2]
                if j+1 < k:
                    heapq.heappush(heap, [-(a[i]-b[j]+b[j+1]), i, j+1])
            return index_new

        def klargest_phone(det, k):
            det_ind = np.argsort(det, axis=1)[:, ::-1]
            det = np.sort(det, axis=1)[:, ::-1]
            index = [[det_ind[0][i]] for i in range(k)]
            for i in range(1, det.shape[0]):
                index = merge(det[0], det[i], det_ind[i], index, k)
            return index

        # print 'weights:'
        # print weights
        # print 'length_weights:'
        # print length_weights

        # gt_roidb = self.gt_roidb()
        gt_roidb = roidb
        assert len(all_boxes) == len(gt_roidb)
        assert len(all_boxes[1]) == 13

        all_length = [0] * 8
        each_length = [0] * 8
        phone_right_each = [0] * 8
        tp, num, tlength, phone_right = 0, 0, 0, 0
        tp_arr = np.zeros((12, ))
        all_arr = np.zeros((12, ))
        extra_tp, extra_num = 0, 0
        res_all = []
        phone_right_list = []
        mat = np.zeros((10, 11), dtype=np.int32)
        test_image_num = 0
        # dis_list = []
        dis_list = [0,0,0,0,0]
        precision_all, recall_all = 0, 0
        for gt, det in zip(gt_roidb, all_boxes):

            gt_labels = gt['labels']
            all_length[gt_labels.shape[0]-5] += 1
            phone_length = np.argmax(safe_log(det[-1][0]) - safe_log(length_weights)) + 5

            det = np.vstack((det[:phone_length]))
            det = safe_log(det[:, :-1])
            det = det - safe_log(weights[:phone_length])

            res = None
            if cfg.TEST.CANDIDATE == 'all':
                res = klargest_phone(det, 2)
                res_all.append(res)
            elif cfg.TEST.CANDIDATE == 'single':
                det_probs_2 = get_labels_rescaling_2(det)[1]
                det_labels_2 = get_labels_rescaling_2(det)[0]
                det_labels_2_rectify = [[det_labels_2[i][0]] if det_probs_2[i][0] > np.log(0.98) else det_labels_2[i] for i in range(len(det_probs_2))]

                res = get_possible_phone(det_labels_2_rectify)
                res_all.append(res)
            elif cfg.TEST.CANDIDATE == 'zero':
                res = [get_labels_rescaling(det)[0]]
                res_all.append(res)

            # twmp
            my_pred = res[0]  # should not change my_pred here
            # print gt_labels
            my_gt = gt_labels.tolist()
            dis = get_dis(my_pred, my_gt)
            dis_list[dis] += 1
            precision, recall = get_measure(my_pred, my_gt)
            precision_all += precision
            recall_all += recall
            # dis_list.append([my_pred, my_gt, dis, precision, recall])


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
                each_length[gt_labels.shape[0]-5] += 1
            
            res1 = np.array(res[0])
            num += gt_labels.shape[0]
            tp += np.sum((res1 == gt_labels))
            tp_arr[np.where(res1 == gt_labels)] += 1
            all_arr[np.arange(gt_labels.shape[0])] += 1
            for i, j in zip(gt_labels, res1):
                mat[i][j] += 1
            if if_phone_right(res, gt_labels):
                phone_right += 1
                phone_right_list.append(test_image_num)
                phone_right_each[gt_labels.shape[0]-5] += 1
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
        print 'right digits: {} / {} = {:.4f}'.format(tp, num, float(tp) / float(num))
        print 'right length: {} / {} = {:.4f}'.format(tlength, len(gt_roidb), float(tlength) / float(len(gt_roidb)))
        print 'right phones: {} / {} = {:.4f}'.format(phone_right, len(gt_roidb), float(phone_right) / float(len(gt_roidb)))
        print 'extra right phones: {:.4f}'.format(float(extra_tp) / float(extra_num))

        # print 'all: {}'.format(all_length)
        # print 'length right: {}'.format(each_length)
        # print 'phone right: {}'.format(phone_right_each)

        # print tp_arr
        # print all_arr
        # print tp_arr.astype(np.float32) / all_arr

        # cache_file = os.path.join(output_dir, 'dis.pkl')
        # with open(cache_file, 'wb') as fid:
        #     cPickle.dump(dis_list, fid, cPickle.HIGHEST_PROTOCOL)
        print("distance: ", dis_list)
        print("distance%: ", [item/float(len(gt_roidb)) for item in dis_list])
        print("mean precision: {} / {} = {:.4f}".format(precision_all, len(gt_roidb), precision_all / len(gt_roidb)))
        print("mean recall: {} / {} = {:.4f}".format(recall_all, len(gt_roidb), recall_all / len(gt_roidb)))

    def evaluate_ohem(self, all_boxes, output_dir, roidb):
        gt_roidb = roidb
        assert len(all_boxes) == len(gt_roidb)
        assert len(all_boxes[1]) == 13

        tp, num, tlength, phone_right = 0, 0, 0, 0
        alpha = 0
        alpha_list = []
        res_all = []
        phone_right_list = []
        # mat = np.zeros((10, 11), dtype=np.int32)
        test_image_num = 0
        for gt, det in zip(gt_roidb, all_boxes): 
            gt_labels = gt['labels']
            phone_length = np.argmax(safe_log(det[-1][0])) + 5
            det = np.vstack((det[:phone_length]))
            det = safe_log(det[:, :-1])
            res = np.argmax(det, axis=1).tolist()
            res_all.append(res)

            if len(res) > gt_labels.shape[0]:
                res = res[:gt_labels.shape[0]] + [len(res)]
            elif len(res) < gt_labels.shape[0]:
                res = res + [10] * (gt_labels.shape[0] - len(res)) + [len(res)]
            else:
                res += [len(res)]
                tlength += 1

            res = np.array(res)
            num += gt_labels.shape[0]
            tp += np.sum((res[:-1] == gt_labels))

            ## the result is slower than evaluate_detections
            if (res[:-1] == gt_labels).all():
                phone_right += 1
                phone_right_list.append(test_image_num)
            test_image_num += 1

            alpha = np.sum((res[:-1] == gt_labels))
            if res[-1]==gt_labels.shape[0]:
                alpha += 1
            alpha = gt_labels.shape[0] + 1 - alpha
            alpha_list.append(alpha + 1)
            # wrong_num+1

        cache_file = os.path.join(output_dir, 'detection_phone.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(res_all, fid, cPickle.HIGHEST_PROTOCOL)
        cache_file = os.path.join(output_dir, 'phone_right_list.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(phone_right_list, fid, cPickle.HIGHEST_PROTOCOL)
        print 'right digits: {} / {} = {:.4f}'.format(tp, num, float(tp) / float(num))
        print 'right length: {} / {} = {:.4f}'.format(tlength, len(gt_roidb), float(tlength) / float(len(gt_roidb)))
        print 'right phones: {} / {} = {:.4f}'.format(phone_right, len(gt_roidb), float(phone_right) / float(len(gt_roidb)))

        alpha_list = np.array(alpha_list).astype(np.float64)
        return alpha_list

