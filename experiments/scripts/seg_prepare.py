import numpy as np
import os
import cPickle
import PIL.Image
from progressbar import ProgressBar
from argparse import ArgumentParser

## 0: background
## 1: 0
## 10: 9
def main(args):
    if not os.path.isdir(os.path.join(args.output_dir)):
        os.makedirs(os.path.join(args.output_dir))

    namelist_path = os.path.join(args.root_dir, args.namelist_name)
    if os.path.exists(namelist_path):
        with open(namelist_path, 'rb') as fid:
            namelist = cPickle.load(fid)
    else:
        raise Exception('No namelist.pkl, init error')

    info_path = os.path.join(args.root_dir, args.info_name)
    if os.path.exists(info_path):
        with open(info_path, 'rb') as fid:
            info = cPickle.load(fid)
    else:
        raise Exception('No info.pkl, init error')

    pbar = ProgressBar(maxval=len(namelist))
    pbar.start()
    i = 0
    for im_name in namelist:
        img = PIL.Image.open(os.path.join(args.input_dir, im_name))
        img = np.array(img, dtype=np.uint8)
        img_seg = np.zeros(img.shape)
        info_im = info[im_name]
        labels, num_bbox = info_im[0], info_im[3]
        for label, bbox in zip(labels, num_bbox):
            x1, y1, x2, y2 = bbox
            img_seg[y1:y2, x1:x2] = 255
            img_seg[y1+2:y2-2, x1+2:x2-2] = label + 1
        img_seg = PIL.Image.fromarray(np.uint8(img_seg))
        img_seg.save(os.path.join(args.output_dir, im_name.split('.')[0]+'.png'), 'png')
        i += 1
        pbar.update(i)
    pbar.finish()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_dir', default='data/express/pretrain_db_benchmark')
    parser.add_argument('--info_name', default='info.pkl')
    parser.add_argument('--namelist_name', default='namelist.pkl')
    parser.add_argument('--input_dir', default='data/express/pretrain_db_benchmark/images')
    parser.add_argument('--output_dir', default='data/express/pretrain_db_benchmark/segment')
    args = parser.parse_args()
    main(args)