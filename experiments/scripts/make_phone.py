import numpy as np
import os
import cPickle
import cv2
import random
from progressbar import ProgressBar

class MakePhone(object):

    def __init__(self, mnist_path=None, phone_path=None):
        self._mnist_path = mnist_path
        self._output_phone_path = phone_path
        self._im_height = 48
        self._processed_image_num = 0
        self._mnist_im_path = os.path.join(self._mnist_path, 'images')
        self._namelist = self._load_namelist()
        self._info = self._load_info()
        self._name_all = []
        self._info_all = {}
        self._shuffle_inds()

    def _load_namelist(self):
        namelist_path = os.path.join(self._mnist_path, 'namelist.pkl')
        if os.path.exists(namelist_path):
            with open(namelist_path, 'rb') as fid:
                return cPickle.load(fid)
        else:
            raise Exception('No namelist.pkl, init error')

    def _load_info(self):
        info_path = os.path.join(self._mnist_path, 'info.pkl')
        if os.path.exists(info_path):
            with open(info_path, 'rb') as fid:
                return cPickle.load(fid)
        else:
            raise Exception('No info.pkl, init error')

    def _shuffle_inds(self):
        self._perm = np.random.permutation(np.arange(len(self._namelist)))
        self._cur = 0

    def _get_next_num_inds(self, size):
        if self._cur + size >= len(self._namelist):
            self._shuffle_inds()

        db_inds = self._perm[self._cur:self._cur + size]
        self._cur += size
        return db_inds

    def _get_next_phone(self, size):
        
        db_inds = self._get_next_num_inds(size)
        namelist_db = [self._namelist[i] for i in db_inds]
        self._get_minibatch(namelist_db)

    def _get_minibatch(self, namelist_db):
        processed_ims = []
        labels = []
        for im_name in namelist_db:
            path = os.path.join(self._mnist_im_path, im_name)
            im = cv2.imread(path)

            try:
                im = self._prep_im(im)
            except:
                print 'im shape error: {}'.format(im_name)
            processed_ims.append(im)
            labels.append(self._info[im_name])
        phone_im = np.concatenate(processed_ims, axis=1)
        phone_label = np.array(labels)

        filename = '{:06d}.jpg'.format(self._processed_image_num)
        cv2.imwrite(os.path.join(self._output_phone_path, 'images', filename), phone_im)
        self._name_all.append(filename)
        self._info_all[filename] = [phone_label, np.array([0, 0, phone_im.shape[1], phone_im.shape[0]])]
        self._processed_image_num += 1

    def _prep_im(self, im):
        im_shape = im.shape

        im_scale = float(self._im_height) / float(im_shape[0])
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        return im

    def forward(self):
        print 'start process'
        phone_num_needed = 4000
        pbar = ProgressBar(maxval=phone_num_needed)
        pbar.start()
        for i in range(phone_num_needed):
            for size in range(5, 13):
                self._get_next_phone(size)
            pbar.update(i)
        pbar.finish()

        random.shuffle(self._name_all)
        cache_file = self._output_phone_path +  'info.pkl'
        with open(cache_file, 'wb') as fid:
            cPickle.dump(self._info_all, fid, cPickle.HIGHEST_PROTOCOL)
        cache_file = self._output_phone_path +  'namelist.pkl'
        with open(cache_file, 'wb') as fid:
            cPickle.dump(self._name_all, fid, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    mnist_path = '/home/sy/code/re_id/express/data/express/pretrain_mnist/'
    phone_path = '/home/sy/code/re_id/express/data/express/makephone/'
    if not os.path.isdir(os.path.join(phone_path, 'images')):
        os.makedirs(os.path.join(phone_path, 'images'))

    random.seed(1024)
    process = MakePhone(mnist_path=mnist_path, phone_path=phone_path)
    process.forward()
