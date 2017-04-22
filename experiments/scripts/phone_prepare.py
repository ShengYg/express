import numpy as np
import os
import cPickle
import cv2
from argparse import ArgumentParser
from progressbar import ProgressBar

def random_crop(x, y, w, h, length):
    x -= float(w) / length / 4
    y -= float(h) / 16
    w += float(w) / length / 2
    h += float(h) / 8
    return max(int(x), 0), max(int(y), 0), int(w), int(h)

def main(args):
    if not os.path.isdir(os.path.join(args.output_dir, 'images')):
        os.makedirs(os.path.join(args.output_dir, 'images'))

    meta = {}
    name_all = []
    path = '/home/sy/code/re_id/express/data/express/'

    namelist_path = path + 'namelist_4.pkl'
    if os.path.exists(namelist_path):
        with open(namelist_path, 'rb') as fid:
            namelist = cPickle.load(fid)
    else:
        raise Exception('No namelist.pkl, init error')

    info_path = path + 'info_4.pkl'
    if os.path.exists(info_path):
        with open(info_path, 'rb') as fid:
            info = cPickle.load(fid)
    else:
        raise Exception('No info.pkl, init error')

    # samples
    print 'start preprocessing'
    pbar = ProgressBar(maxval=len(namelist))
    pbar.start()
    img_num = 0
    i = 0
    for im_name in namelist:
        im = cv2.imread(os.path.join(args.root_dir, im_name))
        info_im = info[im_name]
        boxes, labels = info_im[0], info_im[1]
        # boxes, labels = info_im[2], info_im[4]
        for box, label in zip(boxes, labels):
            if label.shape[0] < 5 or label.shape[0] > 12:
                continue
            x, y, w, h = box
            x, y, w, h = random_crop(x, y, w, h, label.shape[0])
            cropped = im[y:y+h+1, x:x+w+1, :]
            filename = '{:05d}.jpg'.format(img_num)
            cv2.imwrite(os.path.join(args.output_dir, 'images', filename), cropped)

            meta[filename] = label
            name_all.append(filename)
            img_num += 1
        i += 1
        pbar.update(i)
    pbar.finish()
    print 'image nums: {}'.format(i)

    cache_file = os.path.join(args.output_dir, 'namelist.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(name_all, fid, cPickle.HIGHEST_PROTOCOL)

    cache_file = os.path.join(args.output_dir, 'info.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(meta, fid, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='data/express/dataset')
    parser.add_argument('--output_dir', default='data/express/pretrain_db_benchmark_new')
    args = parser.parse_args()
    main(args)