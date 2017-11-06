#!/usr/bin/env python

import collections

import numpy as np
import PIL
import torch
import os
import cPickle
from torch.utils import data


class PhoneSegBase(data.Dataset):

    class_names = np.array([
        'background',
        'zero',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine',
    ])
    # BGR
    # mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    mean_bgr = np.array([104.00698793])

    def __init__(self, root, split='train', transform=False, resize=False):
        self.root = root
        self.split = split
        self._transform = transform
        self._resize_shape = (640, 128)  # (w, h)
        self.resize = resize

        dataset_dir = os.path.join(self.root, 'images')
        segment_dir = os.path.join(self.root, 'segment')
        namelist_path = os.path.join(self.root, 'namelist.pkl')
        namelist = None
        if os.path.exists(namelist_path):
            with open(namelist_path, 'rb') as fid:
                namelist = cPickle.load(fid)
        train_num = int(len(namelist) * 0.8)

        self.files = collections.defaultdict(list)
        if self.split == 'train':
            namelist = namelist[:train_num]
        else:
            namelist = namelist[train_num:]
        
        for file in namelist:
            img_file = os.path.join(dataset_dir, file)
            lbl_file = os.path.join(segment_dir, file)
            self.files[self.split].append({
                'img': img_file,
                'lbl': lbl_file,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        if self.resize:
            img = PIL.Image.Image.resize(img, self._resize_shape)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        if self.resize:
            lbl = PIL.Image.Image.resize(lbl, self._resize_shape)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img[np.newaxis, ]
        # img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img.reshape(img.shape[:-1])
        lbl = lbl.numpy()
        return img, lbl


class PhoneSeg(PhoneSegBase):

    def __init__(self, root, split='train', transform=False, resize=False):
        self.root = root
        self.split = split
        self._transform = transform
        self._resize_shape = (640, 128)  # (w, h)
        self.resize = resize

        dataset_dir = os.path.join(self.root, 'images')
        segment_dir = os.path.join(self.root, 'segment')
        namelist_path = os.path.join(self.root, 'namelist.pkl')
        namelist = None
        if os.path.exists(namelist_path):
            with open(namelist_path, 'rb') as fid:
                namelist = cPickle.load(fid)
        train_num = int(len(namelist) * 0.8)

        self.files = collections.defaultdict(list)
        if self.split == 'train':
            namelist = namelist[:train_num]
        else:
            namelist = namelist[train_num:]
        
        for file in namelist:
            img_file = os.path.join(dataset_dir, file)
            lbl_file = os.path.join(segment_dir, file.split('.')[0]+'.png')
            self.files[self.split].append({
                'img': img_file,
                'lbl': lbl_file,
            })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        if self.resize:
            img = PIL.Image.Image.resize(img, self._resize_shape)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        if self.resize:
            lbl = PIL.Image.Image.resize(lbl, self._resize_shape)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl
