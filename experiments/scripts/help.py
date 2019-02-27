import random
import cPickle
import numpy as np
import os
from PIL import Image
from progressbar import ProgressBar


if __name__ == '__main__':
    # random.seed(2017)
    # namelist = []
    # path_list = ['./data/express/20170509.txt', './data/express/20170510.txt']

    # # # gt.txt -> gt.pkl
    # gt = {}
    # for path in path_list:
    #     print "reading {}".format(path)
    #     with open(path) as f:
    #         lines = f.readlines()
    #     for line in lines:
    #         a = line.split()[0]
    #         b = line.split()[1]
    #         c = line.split()[2]
    #         if b!='null' and c!='null':
    #             b = np.array([int(s) for s in list(b) if s != '-'])
    #             c = np.array([int(s) for s in list(c) if s != '-'])
    #             gt[a] = [b, c]
    
    # with open('./data/express/gt.pkl', 'wb') as fid:
    #     cPickle.dump(gt, fid, cPickle.HIGHEST_PROTOCOL)

    # tif -> jpg
    path1 = '/home/sheng/code/express/data/express/tif/'
    path2 = '/home/sheng/code/express/data/express/jpg/'
    filelist = os.listdir(path1)
    pbar = ProgressBar(maxval=len(filelist))
    pbar.start()
    i = 0
    for filename in filelist:
        im = Image.open(path1+filename)
        im.thumbnail(im.size)
        im = im.convert("RGB")
        im.save(path2+filename.split('.')[0] + '.jpg', 'jpeg', quality=95)
        i += 1
        pbar.update(i)
    pbar.finish()
