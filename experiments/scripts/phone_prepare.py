import numpy as np
import os
import cPickle
import cv2
import random
from progressbar import ProgressBar

RANDOM_CROP = True
NUM_BOX_AVE = True

# version 1
# x -= float(w) / length / 4
# y -= float(h) / 16
# w += float(w) / length / 2
# h += float(h) / 8

def random_crop(x, y, w, h, length):
    x -= float(w) / length * 2 
    y -= float(h) / 16
    w += float(w) / length * 4
    h += float(h) / 8
    return max(int(x), 0), max(int(y), 0), int(w), int(h)

def main(root_dir, info_name, namelist_name, output_dir, prepare):
    if not os.path.isdir(os.path.join(output_dir, 'images')):
        os.makedirs(os.path.join(output_dir, 'images'))

    meta = {}
    meta_refining_det= {}
    name_all = []
    path = './data/express/'

    namelist_path = path + namelist_name
    if os.path.exists(namelist_path):
        with open(namelist_path, 'rb') as fid:
            namelist = cPickle.load(fid)
    else:
        raise Exception('No namelist.pkl, init error')

    info_path = path + info_name
    if os.path.exists(info_path):
        with open(info_path, 'rb') as fid:
            info = cPickle.load(fid)
    else:
        raise Exception('No info.pkl, init error')

    # samples
    if prepare == 'phone':
        print 'start preprocessing phone'
        pbar = ProgressBar(maxval=len(namelist))
        pbar.start()
        i = 0
        for im_name in namelist:
            im = cv2.imread(os.path.join(root_dir, im_name), 0)
            info_im = info[im_name]
            boxes, labels, im_size, num_boxes = info_im
            img_num = 0
            for box, label, num_box in zip(boxes, labels, num_boxes):
                if label.shape[0] < 5 or label.shape[0] > 12:
                    continue
                x, y, w, h = box
                filename = '{}_{}.jpg'.format(im_name[:12], img_num)
                cropped, bbox =  None, None
                sort_ind = np.argsort(num_box[:,0], axis=0)
                num_box = num_box[sort_ind, :]
                if NUM_BOX_AVE:
                    x1 = num_box[1:, :1]
                    x2 = num_box[:-1, 2:3]
                    ind = np.where(x2 - x1 > -10)[0]
                    ave = (x1 + x2) / 2
                    num_box[ind, 2:3] = ave[ind]
                    num_box[ind+1, :1] = ave[ind]
                if RANDOM_CROP:
                    x1, y1, w1, h1 = random_crop(x, y, w, h, label.shape[0])
                    assert x >= x1, '{}, {}'.format(im_name, box)
                    cropped = im[y1:y1+h1+1, x1:x1+w1+1]
                    bbox = np.array([x-x1, y-y1, w, h])
                    sub = np.array([[x1, y1, x1, y1]])
                    num_box = num_box[:, :4] - sub
                else:
                    x1, y1, w1, h1 = x, y, w, h
                    assert x >= x1, '{}, {}'.format(im_name, box)
                    cropped = im[y1:y1+h1+1, x1:x1+w1+1]
                    bbox = np.array([x-x1, y-y1, w, h])
                    sub = np.array([[x1, y1, x1, y1]])
                    num_box = num_box[:, :4] - sub
                    
                a = num_box[:, 2] - num_box[:, 0]
                try:
                    assert np.all(a>0)
                except:
                    print '{}'.format(filename)
                cv2.imwrite(os.path.join(output_dir, 'images', filename), cropped)
                meta[filename] = [label, bbox, im_size, num_box]

                name_all.append(filename)
                img_num += 1
            i += 1
            pbar.update(i)
        pbar.finish()
        print 'express image nums: {}'.format(i)

        random.shuffle(name_all)
        print 'phone image nums: {}'.format(len(name_all))
        cache_file = os.path.join(output_dir, 'namelist.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(name_all, fid, cPickle.HIGHEST_PROTOCOL)

        cache_file = os.path.join(output_dir, 'info.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(meta, fid, cPickle.HIGHEST_PROTOCOL)

    elif prepare == 'num':
        print 'start preprocessing phone num'
        pbar = ProgressBar(maxval=len(namelist))
        pbar.start()
        img_num = 0
        j = 0
        for im_name in namelist:
            im = cv2.imread(os.path.join(root_dir, im_name), 0)
            info_im = info[im_name]
            boxes, labels = info_im[3], info_im[1]
            for box, label in zip(boxes, labels):
                if label.shape[0] < 5 or label.shape[0] > 12:
                    continue
                assert box.shape[0] == label.shape[0], 'num boxes error!'
                for i in range(box.shape[0]):
                    x1, y1, x2, y2 = box[i][:4]
                    cropped = im[y1:y2, x1:x2]
                    filename = '{:06d}.jpg'.format(img_num)
                    cv2.imwrite(os.path.join(output_dir, 'images', filename), cropped)

                    meta[filename] = label[i]
                    name_all.append(filename)
                    img_num += 1
            j += 1
            pbar.update(j)
        pbar.finish()
        print 'image nums: {}'.format(j)

        random.shuffle(name_all)
        cache_file = os.path.join(output_dir, 'namelist.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(name_all, fid, cPickle.HIGHEST_PROTOCOL)

        cache_file = os.path.join(output_dir, 'info.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(meta, fid, cPickle.HIGHEST_PROTOCOL)

    elif prepare == 'phone_extra':
        print 'start preprocessing extra phone'
        pbar = ProgressBar(maxval=len(namelist))
        pbar.start()
        i = 0
        for im_name in namelist:
            im = cv2.imread(os.path.join(root_dir, im_name), 0)
            info_im = info[im_name]
            boxes, labels, im_size, num_boxes = info_im
            img_num = 0
            for box, label, num_box in zip(boxes, labels, num_boxes):
                name_all_tmp = []
                if label.shape[0] < 5 or label.shape[0] > 12:
                    continue

                x, y, w, h = box
                cropped, bbox =  None, None
                sort_ind = np.argsort(num_box[:,0], axis=0)
                num_box = num_box[sort_ind, :]

                x1, y1, w1, h1 = random_crop(x, y, w, h, label.shape[0])
                assert x >= x1, '{}, {}'.format(im_name, box)
                cropped = im[y1:y1+h1+1, x1:x1+w1+1]
                bbox = np.array([x-x1, y-y1, w, h])
                sub = np.array([[x1, y1, x1, y1]])
                num_box = num_box[:, :4] - sub

                a = num_box[:, 2] - num_box[:, 0]
                try:
                    assert np.all(a>0)
                except:
                    print '{}'.format(filename)

                filename = '{}_{}.jpg'.format(im_name[:12], img_num)
                cv2.imwrite(os.path.join(output_dir, 'images', filename), cropped)
                meta[filename] = [label, bbox, im_size, num_box]
                name_all_tmp.append(filename)
                img_num += 1

                if label.shape[0] == 11:
                    for phone_length in range(5, 11):
                        start = random.randint(0, 11 - phone_length)
                        end = start + phone_length - 1

                        x_in = num_box[start][0] + x1
                        w_in = num_box[end][2] - num_box[start][0]

                        cropped = np.concatenate((im[y1:y1+h1, x1:x], im[y1:y1+h1, x_in:x_in+w_in], im[y1:y1+h1, x+w:x1+w1]), axis=1) 
                        bbox = np.array([x-x1, y-y1, w_in, h])

                        filename = '{}_{}.jpg'.format(im_name[:12], img_num)
                        cv2.imwrite(os.path.join(output_dir, 'images', filename), cropped)

                        meta[filename] = [label[start:end+1], bbox, im_size, num_box[start:end+1]]
                        name_all_tmp.append(filename)
                        img_num += 1
                name_all.append(name_all_tmp)
            i += 1
            pbar.update(i)
        pbar.finish()
        print 'express image nums: {}'.format(i)

        if prepare != 'phone_extra':
            random.shuffle(name_all)
        else:
            random.shuffle(name_all)
            name_all = reduce(lambda x,y :x+y ,name_all)

        print 'phone image nums: {}'.format(len(name_all))
        cache_file = os.path.join(output_dir, 'namelist.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(name_all, fid, cPickle.HIGHEST_PROTOCOL)

        cache_file = os.path.join(output_dir, 'info.pkl')
        with open(cache_file, 'wb') as fid:
            cPickle.dump(meta, fid, cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # # mnist
    root_dir = 'data/express/dataset'
    info_name = 'info_phone.pkl'
    namelist_name = 'namelist_phone.pkl'
    output_dir = 'data/express/pretrain_mnist'
    prepare = 'num'
    random.seed(1024)
    main(root_dir, info_name, namelist_name, output_dir, prepare)

    # # train phone
    root_dir = 'data/express/dataset'
    info_name = 'info_phone.pkl'
    namelist_name = 'namelist_phone.pkl'
    output_dir = 'data/express/pretrain_db_benchmark'
    prepare = 'phone'
    random.seed(1024)
    main(root_dir, info_name, namelist_name, output_dir, prepare)

    # # test phone
    root_dir = 'data/express/dataset'
    info_name = 'info_test.pkl'
    namelist_name = 'namelist_test.pkl'
    output_dir = 'data/express/test_db_benchmark'
    prepare = 'phone'
    random.seed(1024)
    main(root_dir, info_name, namelist_name, output_dir, prepare)

    # train extra phone
    # root_dir = 'data/express/dataset'
    # info_name = 'info_phone.pkl'
    # namelist_name = 'namelist_phone.pkl'
    # output_dir = 'data/express/pretrain_db_benchmark_extra'
    # prepare = 'phone_extra'
    # random.seed(1024)
    # main(root_dir, info_name, namelist_name, output_dir, prepare)