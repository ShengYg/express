import xml.etree.ElementTree as ET
import os
import numpy as np
import cPickle
import random
from progressbar import ProgressBar
import time
from PIL import Image

"""
remove replicated image before run data_prepare
"""

PHONE_WIDTH = 12

def outbox(box): # (x1, y1, x2, y2)
    x = min(box[:, 0])
    y = min(box[:, 1])
    w = max(box[:, 2])-min(box[:, 0])
    h = max(box[:, 3])-min(box[:, 1])
    return np.array([[x, y, w, h]])


def get_phonenum(box):
    return box[:, 4]

def get_phonenum_list(box):
    return list(box[:, 4])

def get_box(filename):
    if not os.path.exists(filename):
        print 'name error'
    try:
        tree = ET.ElementTree(file = filename)
    except:
        raise
    box_all = np.array([0,0,0,0,0])
    for elem in tree.iter():
        if elem.tag == 'Result':
            info = elem.attrib
            if info['ResultType'] == "4":
                break
        if elem.tag == 'markinfo':
            info = elem.attrib
            if info['RealValue'] != '-':
                box_all = np.vstack((box_all, np.array([int(info['x']), int(info['y']), int(info['w']), int(info['h']), int(info['RealValue'][0])])))
    return box_all[1:] if len(box_all.shape) != 1 else np.array([[0,0,0,0,0]])


def get_box_4(filename):
    if not os.path.exists(filename):
        print 'name error'
    try:
        tree = ET.ElementTree(file = filename)
    except:
        raise
    box_all = []
    boxes = np.array([0,0,0,0,0])
    for elem in tree.iter():
        if elem.tag == 'Result':
            info = elem.attrib
            if info['ResultType'] == "2":
                break
        if elem.tag == 'markinfo':
            info = elem.attrib
            if info['RealValue'] != '-':
                if info['RealValue'] == '.':
                    if len(boxes.shape) != 1:
                        box_all.append(boxes[1:])
                        boxes = np.array([0,0,0,0,0])
                else:
                    boxes = np.vstack((boxes, np.array([int(info['x']), int(info['y']), int(info['w']), int(info['h']), int(info['RealValue'][0])])))
    if len(boxes.shape) != 1:
        box_all.append(boxes[1:])
    return box_all
    

def get_phone_box(box):
    '''
    return: phone_num, phone_box
    '''
    ave_width = sum(box[:, 2]) / box.shape[0]
    box[:, 2] += box[:, 0]
    box[:, 3] += box[:, 1]

    if box.shape[0] > PHONE_WIDTH:  # horizontal cut
        x_dis = abs(np.hstack((box[1:, 0], box[-1,0])) - box[:, 0])
        ind = np.where(x_dis > ave_width * 2)[0] + 1
        boxes = np.split(box, ind)
        phonenum = map(get_phonenum, boxes)
        return (len(boxes), np.vstack(map(outbox, boxes)), phonenum)
    else:
        phonenum = [get_phonenum(box)]
        return (1, outbox(box), phonenum)


def split_box_vertical(box):
    '''
    return: phone_num, phone_box
    '''
    if max(box[:, 1]) - min(box[:, 1]) < box_two_thres:
        phones, box_phone, phone_label = get_phone_box(box)
        return phones, box_phone, phone_label
    else:
        y_dis = abs(box[:, 1] - box[0, 1])
        inds1 = np.where(y_dis < box_height_thres)[0]
        inds2 = np.where(y_dis >= box_height_thres)[0]
        phones1, box_phone1, phone_label1 = split_box_vertical(box[inds1])
        phones2, box_phone2, phone_label2 = split_box_vertical(box[inds2])
        return phones1 + phones2, np.vstack((box_phone1, box_phone2)), phone_label1 + phone_label2

def phone_match(old, new):
    ret = []
    for item1 in old:
        match = False
        for item2 in new:
            if item1.shape[0] == item2.shape[0] and (item1 == item2).all():
                match = True
                break
            elif item1.shape[0] > item2.shape[0] and (item1[item1.shape[0] - item2.shape[0]:] == item2).all():
                match = True
                break
        ret.append(match)
    return ret

def phone_len_match(label, gt):
    ret = []
    gt_flag = [False] * len(gt)
    for item1 in label:
        match = False
        for j in range(len(gt)):
            if not gt_flag[j] and item1.shape[0] == gt[j].shape[0]:
                match = True
                gt_flag[j] = True
                break
            # elif item1.shape[0] > gt[j].shape[0] and (item1[item1.shape[0] - gt[j].shape[0]:] == gt[j]).all():
            #     match = True
            #     break
        ret.append(match)
    return ret

def main(dataset_path, info_path, namelist_path, namelist_type=0, exclude_namelist=None):
    ResultType = '4'
    box_two_thres = 60
    box_height_thres = 30
    info_all = {}
    namelist = []
    if not os.path.isdir(dataset_path):
        print 'error data dir path {}'.format(dataset_path)
    filelist = sorted(os.listdir(dataset_path))
    if not filelist:
        print 'no pic in dir {}'.format(dataset_path)

    cache_file = './data/express/gt.pkl'
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as fid:
            gt_phone = cPickle.load(fid)
    else:
        raise Exception('No gt.pkl')

    exclude = []
    if exclude_namelist:
        print 'exclude file exist'
        if os.path.exists(exclude_namelist):
            with open(exclude_namelist, 'rb') as fid:
                exclude = cPickle.load(fid)
    
    if ResultType == '2':
        print 'ResultType: {}'.format(ResultType)
        pbar = ProgressBar(maxval=len(filelist))
        pbar.start()
        i = 0
        print 'process start'
        for filename in filelist:
            num, suffix = filename.split('.')[0], filename.split('.')[1]
            if suffix == 'xml':
                try:
                    box_num = get_box(path + filename)
                except ET.ParseError:
                    print 'XML exception, filename: {}'.format(filename)
                    pass
                if box_num.shape[0] < 4:
                    continue
                info_all[num + '.jpg'] = [box_num[:, :4]]
                phones, box_phone, phone_label = split_box_vertical(box_num)
                info_all[num + '.jpg'].extend([box_num.shape[0], box_phone, phones, phone_label])

                assert phones == len(phone_label), "init error! nums error"
                assert sum([j.shape[0] for j in phone_label]) == box_num.shape[0], "init error! phones error{}, {}".format(phone_label, box_num.shape)
                namelist.append(num + '.jpg')
                # get img info
                sourcefile = path + num + '.jpg'
                if not os.path.isfile(sourcefile):
                    raise Exception('No file.jpg')
                else:
                    img = Image.open(sourcefile)
                    img_size = img.size
                    info_all[num + '.jpg'].append(img_size)

            i += 1
            pbar.update(i)
        pbar.finish()

        # info_all: [num_box, nums, phone_box, phones, img_size, phone_label]
        cache_file = '/home/sy/code/re_id/express/data/express/info_2.pkl'
        with open(cache_file, 'wb') as fid:
            cPickle.dump(info_all, fid, cPickle.HIGHEST_PROTOCOL)
        cache_file = '/home/sy/code/re_id/express/data/express/namelist_2.pkl'
        with open(cache_file, 'wb') as fid:
            cPickle.dump(namelist, fid, cPickle.HIGHEST_PROTOCOL)
    else:
        print 'ResultType: {}'.format(ResultType)
        pbar = ProgressBar(maxval=len(filelist))
        pbar.start()
        i = 0
        print 'process start'
        for filename in filelist:
            num, suffix = filename.split('.')
            if suffix == 'xml':
                boxes = None
                try:
                    boxes = get_box_4(dataset_path + filename)
                except ET.ParseError:
                    print 'XML exception, filename: {}'.format(filename)
                    pass
                except:
                    print filename

                ## getting all names
                # info [phone_box, phone_label, img_size]
                if namelist_type == 0:
                    phone_box, phone_label = None, None
                    if boxes:
                        go_on = False
                        for box in boxes:
                            box[:, 2] += box[:, 0]
                            box[:, 3] += box[:, 1]
                        phone_box, phone_label = np.vstack(map(outbox, boxes)), map(get_phonenum, boxes)
                        for phone in phone_label:
                            if phone.shape[0] > PHONE_WIDTH:
                                go_on = True
                        if not go_on:
                            namelist.append(num + '.jpg')
                            info_all[num + '.jpg'] = [phone_box, phone_label]

                            sourcefile = dataset_path + num + '.jpg'
                            if not os.path.isfile(sourcefile):
                                raise Exception('No {}'.format(sourcefile))
                            else:
                                img = Image.open(sourcefile)
                                img_size = img.size
                                info_all[num + '.jpg'].append(img_size)

                ## getting label from image and xml, used for training express
                # info [phone_box, phone_label, img_size]
                if namelist_type == 1:
                    phone_box, phone_label = None, None
                    key = num.split('_')[0]
                    if boxes:
                        go_on = False
                        for box in boxes:
                            box[:, 2] += box[:, 0]
                            box[:, 3] += box[:, 1]
                        phone_box, phone_label = np.vstack(map(outbox, boxes)), map(get_phonenum, boxes)
                        for phone in phone_label:
                            if phone.shape[0] > PHONE_WIDTH or phone.shape[0] < 5:
                                go_on = True
                        if not go_on:
                            assert len(phone_label) == phone_box.shape[0], 'pic: {}, lebel: {}, box: {}'.format(num, phone_label, phone_box.shape[0])
                            if len(gt_phone[key]) != len(phone_label):
                                continue

                            # ind = phone_len_match(phone_label, gt_phone[key])
                            # if not (np.array(ind) == True).all():
                            #     continue

                            if num + '.jpg' not in exclude:
                                namelist.append(num + '.jpg')
                                info_all[num + '.jpg'] = [phone_box, phone_label]

                                sourcefile = dataset_path + num + '.jpg'
                                if not os.path.isfile(sourcefile):
                                    raise Exception('No {}'.format(sourcefile))
                                else:
                                    img = Image.open(sourcefile)
                                    img_size = img.size
                                    info_all[num + '.jpg'].append(img_size)

                # getting label from gt_phone, used for training phone
                # it should be chosen carefully to make test_set large
                # info_phone [phone_box, phone_label, img_size, num_box]
                elif namelist_type == 2:
                    phone_box, phone_label = None, None
                    key = num.split('_')[0]
                    if boxes:
                        go_on = False
                        for box in boxes:
                            box[:, 2] += box[:, 0]
                            box[:, 3] += box[:, 1]
                        phone_box, phone_label = np.vstack(map(outbox, boxes)), map(get_phonenum, boxes)
                        for phone in phone_label:
                            if phone.shape[0] > PHONE_WIDTH:
                                go_on = True
                        if not go_on:
                            assert len(phone_label) == phone_box.shape[0], 'pic: {}, lebel: {}, box: {}'.format(num, phone_label, phone_box.shape[0])
                            # phone_label = gt_phone[key]
                            ind = phone_match(phone_label, gt_phone[key])
                            phone_label = [phone_label[j] for j in range(len(phone_label)) if ind[j]]
                            if len(phone_label) <= 0:
                                continue
                            ind_ = [j if ind[j] else -1 for j in range(len(ind))]
                            ind_ = filter(lambda x:x>=0, ind_)
                            phone_box = phone_box[ind_]
                            num_box = [boxes[j] for j in range(len(boxes)) if ind[j]]
                            if num + '.jpg' not in exclude:
                                namelist.append(num + '.jpg')
                                assert phone_box.shape[0] == len(phone_label)
                                info_all[num + '.jpg'] = [phone_box, phone_label]
                                sourcefile = dataset_path + num + '.jpg'
                                if not os.path.isfile(sourcefile):
                                    raise Exception('No {}'.format(sourcefile))
                                else:
                                    img = Image.open(sourcefile)
                                    img_size = img.size
                                    info_all[num + '.jpg'].append(img_size)
                                info_all[num + '.jpg'].append(num_box)

                # getting label from gt_phone, all included ,used for testing all
                # info_phone [phone_box, phone_label, img_size]
                elif namelist_type == 3:
                    phone_box, phone_label = None, None
                    key = num.split('_')[0]
                    if boxes:
                        go_on = False
                        for box in boxes:
                            box[:, 2] += box[:, 0]
                            box[:, 3] += box[:, 1]
                        phone_box, phone_label = np.vstack(map(outbox, boxes)), map(get_phonenum, boxes)
                        for phone in phone_label:
                            if phone.shape[0] > PHONE_WIDTH:
                                go_on = True
                        if not go_on:
                            assert len(phone_label) == phone_box.shape[0], 'pic: {}, lebel: {}, box: {}'.format(num, phone_label, phone_box.shape[0])
                            try:
                                gt_phone[key]
                            except:
                                print 'no key'
                                continue
                            ind = phone_match(phone_label, gt_phone[key])
                            assert len(ind) > 0
                            if (np.array(ind) == False).any():
                                continue
                            if len(ind) != len(gt_phone[key]):
                                continue
                            num_box = [boxes[j] for j in range(len(boxes)) if ind[j]]
                            namelist.append(num + '.jpg')
                            info_all[num + '.jpg'] = [phone_box, phone_label]
                            sourcefile = dataset_path + num + '.jpg'
                            if not os.path.isfile(sourcefile):
                                raise Exception('No {}'.format(sourcefile))
                            else:
                                img = Image.open(sourcefile)
                                img_size = img.size
                                info_all[num + '.jpg'].append(img_size)
                            info_all[num + '.jpg'].append(num_box)

            i += 1
            pbar.update(i)
        pbar.finish()

        # info_all: [phone_box, phone_label, img_size, num_box]
        with open(info_path, 'wb') as fid:
            cPickle.dump(info_all, fid, cPickle.HIGHEST_PROTOCOL)
        random.shuffle(namelist)
        print 'namelist length: {}'.format(len(namelist))
        if namelist_type == 3:
            namelist = namelist[:1400]
            print 'type3: selected namelist length: {}'.format(len(namelist))
        with open(namelist_path, 'wb') as fid:
            cPickle.dump(namelist, fid, cPickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    ## 1. all bbox information
    ## [phone_box, phone_label, img_size]
    dataset_path = 'data/express/dataset/'
    info_path = 'data/express/info.pkl'
    namelist_path = 'data/express/namelist.pkl'
    namelist_type = 0
    exclude_namelist = None
    random.seed(1024)
    main(dataset_path, info_path, namelist_path, namelist_type, exclude_namelist)

    ## 2. some bbox information(bbox right, phone right, choose 1000), for test detection+recognition
    ## [phone_box, phone_label, img_size, num_box]
    dataset_path = 'data/express/dataset/'
    info_path = 'data/express/info_test.pkl'
    namelist_path = 'data/express/namelist_test.pkl'
    namelist_type = 3
    exclude_namelist = None
    random.seed(1024)
    main(dataset_path, info_path, namelist_path, namelist_type, exclude_namelist)

    ## 3. some bbox information(bbox right, exclude namelist_test), for train/test detection
    ## [phone_box, phone_label, img_size]
    dataset_path = 'data/express/dataset/'
    info_path = 'data/express/info_express.pkl'
    namelist_path = 'data/express/namelist_express.pkl'
    namelist_type = 1
    exclude_namelist = 'data/express/namelist_test.pkl'
    random.seed(1024)
    main(dataset_path, info_path, namelist_path, namelist_type, exclude_namelist)

    ## 4. some bbox information(phone right, exclude namelist_test), for train/test recognition
    ## [phone_box, phone_label, img_size, num_box]
    dataset_path = 'data/express/dataset/'
    info_path = 'data/express/info_phone.pkl'
    namelist_path = 'data/express/namelist_phone.pkl'
    namelist_type = 2
    exclude_namelist = 'data/express/namelist_test.pkl'
    random.seed(1024)
    main(dataset_path, info_path, namelist_path, namelist_type, exclude_namelist)
