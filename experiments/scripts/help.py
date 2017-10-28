import random
import cPickle
import numpy as np
import os
from PIL import Image


if __name__ == '__main__':
    random.seed(2017)
    namelist = []
    path = './data/express/20170509.txt'

    # # gt.txt -> gt.pkl
    gt = {}
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        a = line.split()[0]
        b = line.split()[1]
        c = line.split()[2]
        if b!='null' and c!='null':
            b = np.array([int(s) for s in list(b) if s != '-'])
            c = np.array([int(s) for s in list(c) if s != '-'])
            gt[a] = [b, c]
    with open('./data/express/gt.pkl', 'wb') as fid:
        cPickle.dump(gt, fid, cPickle.HIGHEST_PROTOCOL)

    # tif -> jpg
    # path1 = '/home/sy/code/re_id/express/data/express/dataset2/tif/'
    # path2 = '/home/sy/code/re_id/express/data/express/dataset2/jpg/'
    # filelist = os.listdir(path1)
    # # filelist[0]
    # for filename in filelist:
    #     im = Image.open(path1+filename)
    #     im.thumbnail(im.size)
    #     im.save(path2+filename.split('.')[0] + '.jpg', 'jpeg', quality=95)
