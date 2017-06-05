import os
import cPickle
import numpy as np


if __name__ == '__main__':
    filepath = os.path.join(os.getcwd(), 'data', 'express', 'gt.pkl')
    with open(filepath) as fid:
        gt = cPickle.load(fid)

    gt_phone = None
    gt_phone_num = []
    det_phone_num = []
    right_phone_num = []

    det_phone_list = []
    error_check = []
    filename = None
    filepath = os.path.join(os.getcwd(), 'demo', 'out.txt')
    if not os.path.isfile(filepath):
        raise IOError(('{:s} not found.').format(filepath))

    with open(filepath) as fid:
        for line in fid:
            if line[0] != '[':
                if gt_phone:
                    right_phone = 0
                    for gt_i in gt_phone:
                        for det in det_phone_list:
                            det = np.array(det)
                            if (gt_i.shape[0] == det.shape[0]) and (gt_i == det).all():
                                right_phone += 1
                                break
                            elif (gt_i.shape[0] < det.shape[0]) and (gt_i == det[det.shape[0] - gt_i.shape[0]:]).all():
                                right_phone += 1
                                break
                    right_phone_num.append(right_phone)
                    det_phone_num.append(len(det_phone_list))
                    gt_phone_num.append(len(gt_phone))
                    det_phone_list = []

                filename = line.split('_')[0]
                gt_phone = gt[filename]
                
            elif line[0] == '[':
                det_phone = np.array([int(digit) for digit in line[1:-2].split(' ')])
                det_phone_list.append(det_phone)

            else:
                print 'error'
                print line

    right_phone = 0
    for gt_i in gt_phone:
        for det in det_phone_list:
            det = np.array(det)
            if (gt_i.shape[0] == det.shape[0]) and (gt_i == det).all():
                right_phone += 1
                break
            elif (gt_i.shape[0] < det.shape[0]) and (gt_i == det[det.shape[0] - gt_i.shape[0]:]).all():
                right_phone += 1
                break
    right_phone_num.append(right_phone)
    det_phone_num.append(len(det_phone_list))
    gt_phone_num.append(len(gt_phone))

    filepath = os.path.join(os.getcwd(), 'demo', 'error_check.pkl')
    with open(filepath, 'wb') as fid:
        cPickle.dump(error_check, fid, cPickle.HIGHEST_PROTOCOL)

    assert len(gt_phone_num) == len(det_phone_num)
    assert len(gt_phone_num) == len(right_phone_num)
    print 'express: {}'.format(len(gt_phone_num))
    print 'gt:      {}'.format(sum(gt_phone_num))
    print 'det:     {}'.format(sum(det_phone_num))
    print 'right:   {}'.format(sum(right_phone_num))
    print 'acc:     {}'.format(float(sum(right_phone_num)) / sum(gt_phone_num))
