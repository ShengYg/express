import os
import cPickle
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--thres1', dest='thres1', default=0.8, type=float)
    parser.add_argument('--thres2', dest='thres2', default=-1, type=float)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    filelist, all_boxes, score_namelist, score_all, gt = None, None, None, None, None
    cache_file = os.path.join(os.getcwd(), 'demo', 'img_namelist.pkl')
    with open(cache_file, 'rb') as fid:
        filelist = cPickle.load(fid)
    cache_file = os.path.join(os.getcwd(), 'demo', 'img_detections.pkl')
    with open(cache_file, 'rb') as fid:
        all_boxes = cPickle.load(fid)
    cache_file = os.path.join(os.getcwd(), 'demo', 'phone_score.pkl')
    with open(cache_file, 'rb') as fid:
        score_all = cPickle.load(fid)
    cache_file = os.path.join(os.getcwd(), 'data', 'express', 'gt.pkl')
    with open(cache_file, 'rb') as fid:
        gt = cPickle.load(fid)

    all_phone, right_phone, gt_phone_num = 0, 0, 0
    right_phone_list = []
    # 0: no
    # 1: yes
    # 2: pass
    for im_name in filelist:
        name = im_name.split('_')[0]
        det = all_boxes[name]
        det_ind = np.where(det[:, 4] > args.thres1)[0]
        gt_phone = gt[name]
        gt_phone_num += len(gt_phone)
        for i in det_ind:
            crop_img_name = name + '_{}.jpg'.format(i)
            res, score = score_all[crop_img_name]
            if score < args.thres2:
                continue
            all_phone += 1
            for gt_i in gt_phone:
                if (gt_i.shape[0] == res.shape[0]) and (gt_i == res).all():
                    right_phone += 1
                    right_phone_list.append(crop_img_name)
                    # print crop_img_name
                    break
                elif (gt_i.shape[0] < res.shape[0]) and (gt_i == res[res.shape[0] - gt_i.shape[0]:]).all():
                    right_phone += 1
                    right_phone_list.append(crop_img_name)
                    # print crop_img_name
                    break

    print 'all phone num: {}'.format(all_phone)
    print 'gt phone num: {}'.format(gt_phone_num)
    print 'acc: {} / {} = {}'.format(right_phone, gt_phone_num, float(right_phone) / gt_phone_num)
    cache_file = os.path.join(os.getcwd(), 'demo', 'right_phone_list.pkl')
    with open(cache_file, 'wb') as fid:
        cPickle.dump(right_phone_list, fid, cPickle.HIGHEST_PROTOCOL)

