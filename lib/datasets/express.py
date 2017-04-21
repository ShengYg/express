import os
import numpy as np
from scipy.sparse import csr_matrix
import datasets
from datasets.imdb import imdb
from fast_rcnn.config import cfg
import cPickle
from progressbar import ProgressBar
import cv2

def _compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

class express(imdb):
    def __init__(self, image_set, root_dir=None, ratio=0.8):
        super(express, self).__init__('express_' + image_set)
        self._image_set = image_set
        self._training_ratio = ratio
        self._root_dir = self._get_default_path() if root_dir is None \
                         else root_dir
        self._data_path = os.path.join(self._root_dir, 'dataset')
        self._classes = ('__background__', 'phonenum')
        self._namelist = self._load_namelist()
        self._info = self._load_info()
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        assert os.path.isdir(self._root_dir), \
                "express does not exist: {}".format(self._root_dir)
        assert os.path.isdir(self._data_path), \
                "Path does not exist: {}".format(self._data_path)
        self.config = {'rpn_file'    : None}

    def _load_namelist(self):
        namelist_path = os.path.join(self._root_dir, 'namelist_4.pkl')
        if os.path.exists(namelist_path):
            with open(namelist_path, 'rb') as fid:
                return cPickle.load(fid)
        else:
            raise Exception('No namelist.pkl, init error')

    def _load_info(self):
        info_path = os.path.join(self._root_dir, 'info_4.pkl')
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
        train_num = int(len(self._namelist) * 0.8)
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
            pic_info = self._info[index]
            # boxes = pic_info[2]
            # length = pic_info[3]
            # im_size = pic_info[5]
            # assert(boxes.shape[0] == length), 'info data error!'
            boxes = pic_info[0]
            im_size = pic_info[2]
            label = pic_info[1]

            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            num_objs = boxes.shape[0]
            gt_classes = np.ones((num_objs), dtype=np.int32)
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            overlaps[:, 1] = 1.0
            overlaps = csr_matrix(overlaps)
            gt_roidb.append({
                'boxes': boxes,
                'label': label,
                'image': index,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'im_size': im_size})
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print "wrote gt roidb to {}".format(cache_file)
        return gt_roidb

    def _get_default_path(self):
        return os.path.join(cfg.DATA_DIR, 'express')

    def evaluate_detections(self, all_boxes, output_dir, iou_thres=0.6):
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
            # # add position info
            im_size = gt['im_size']
            inds = np.where(det[:, 0] < im_size[0]/2)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            if num_det == 0:
                count_gt += num_gt
                continue
            ious = np.zeros((num_gt, num_det), dtype=np.float32)
            for i in xrange(num_gt):
                for j in xrange(num_det):
                    ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
            tfmat = (ious >= iou_thres)
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
        print 'mAP: {:.4%}'.format(ap)

    def get_detections(self, all_boxes, output_dir, iou_thres=0.5):
        # all_boxes[cls][image] = N x 5 (x1, y1, x2, y2, score)
        cache_file = os.path.join(self._root_dir, 'test_all_db', 'namelist.pkl')
        if os.path.exists(cache_file):
            print '###### already get detections'
            pass
        else:
            outdir = os.path.join(self._root_dir, 'test_all_db', 'images')
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            gt_roidb = self.gt_roidb()
            print 'gt image nums: {}'.format(len(gt_roidb))
            assert len(all_boxes) == 2
            assert len(all_boxes[1]) == len(gt_roidb)

            pbar = ProgressBar(maxval=len(gt_roidb))
            pbar.start()
            img_num = 0
            k = 0
            meta = {}
            name_all = []
            for gt, det in zip(gt_roidb, all_boxes[1]):
                gt_boxes = gt['boxes']
                det = np.asarray(det)
                # # add position info
                im_width = gt['im_size'][0]
                im_name = gt['image']
                im_label = gt['label']
                inds = np.where(det[:, 0] < im_width/2)[0]
                det = det[inds]
                num_gt = gt_boxes.shape[0]
                num_det = det.shape[0]
                if num_det == 0:
                    continue

                ious = np.zeros((num_gt, num_det), dtype=np.float32)
                for i in xrange(num_gt):
                    for j in xrange(num_det):
                        ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])

                # for each gt, keep only the largest iou of all the det
                out_det = np.zeros((num_gt, 4))
                for i in xrange(num_gt):
                    largest_ind = np.argmax(ious[i, :])
                    if ious[i, largest_ind] < iou_thres:
                        out_det[i] = np.array([0, 0, 0, 0])
                    else:
                        out_det[i] = det[largest_ind][:4]
                # out_det = out_det.astype(np.int32, copy=False)

                # crop images
                im = cv2.imread(os.path.join(self._data_path, im_name))
                for box, label in zip(out_det, im_label):
                    if label.shape[0] < 5 or label.shape[0] > 12:
                        continue
                    x1, y1, x2, y2 = box
                    x1 = int(x1 - im_width * 0.000)
                    x2 = int(x2 - im_width * 0.000)
                    y1 = int(y1)
                    y2 = int(y2)
                    if (box == np.array([0, 0, 0, 0])).all():
                        continue
                    # x, y, w, h = random_crop(x, y, w, h, label.shape[0])
                    cropped = im[y1:y2+1, x1:x2+1, :]
                    filename = '{:05d}.jpg'.format(img_num)
                    # if img_num == 9984:
                    #     print box
                    #     print x, y, w, h
                    cv2.imwrite(os.path.join(self._root_dir, 'test_all_db', 'images', filename), cropped)
                    ### preprocess
                    meta[filename] = label
                    name_all.append(filename)
                    img_num += 1
                k += 1
                pbar.update(k)
            pbar.finish()

            cache_file = os.path.join(self._root_dir, 'test_all_db', 'namelist.pkl')
            with open(cache_file, 'wb') as fid:
                cPickle.dump(name_all, fid, cPickle.HIGHEST_PROTOCOL)

            cache_file = os.path.join(self._root_dir, 'test_all_db', 'info.pkl')
            with open(cache_file, 'wb') as fid:
                cPickle.dump(meta, fid, cPickle.HIGHEST_PROTOCOL)

    def get_detection_error(self, all_boxes, iou_thres=0.5):
        # test whether the box is larger or smaller in horizontal
        gt_roidb = self.gt_roidb()
        print 'gt image nums: {}'.format(len(gt_roidb))
        assert len(all_boxes) == 2
        assert len(all_boxes[1]) == len(gt_roidb)

        pbar = ProgressBar(maxval=len(gt_roidb))
        pbar.start()
        error = []
        i = 0
        meta = {}
        for gt, det in zip(gt_roidb, all_boxes[1]):
            gt_boxes = gt['boxes']
            det = np.asarray(det)
            # # add position info
            im_width = gt['im_size'][0]
            inds = np.where(det[:, 0] < im_width/2)[0]
            det = det[inds]
            num_gt = gt_boxes.shape[0]
            num_det = det.shape[0]
            out_det = np.zeros((num_gt, 4))
            if num_det != 0:
                ious = np.zeros((num_gt, num_det), dtype=np.float32)
                for i in xrange(num_gt):
                    for j in xrange(num_det):
                        ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
                
                for i in xrange(num_gt):
                    largest_ind = np.argmax(ious[i, :])
                    if ious[i, largest_ind] < iou_thres:
                        out_det[i] = np.array([0, 0, 0, 0])
                    out_det[i] = det[largest_ind][:4]
                out_det = out_det.astype(np.int32, copy=False)
            out_det = out_det.astype(np.float32, copy=False)
            gt_boxes = gt_boxes.astype(np.float32, copy=False)
            left = (out_det[:, 0] - gt_boxes[:, 0]) / im_width
            right = (out_det[:, 0] + out_det[:, 2] - gt_boxes[:, 0] - gt_boxes[:, 2]) / im_width
            error.extend([[x, y]for x, y in zip(left, right)])

            i += 1
            pbar.update(i)
        pbar.finish()

        print 'gt phone nums: {}'.format(len(error))
        cache_file = os.path.join(self._root_dir, 'test_all_db', 'error.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(error, fid, cPickle.HIGHEST_PROTOCOL)