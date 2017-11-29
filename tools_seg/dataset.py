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

    def __init__(self, root, split='train', transform=False, resize=False, pos=False):
        self.root = root
        self.split = split
        self._transform = transform
        self._resize_shape = (640, 128)  # (w, h)
        self.resize = resize
        self.pos = pos

        dataset_dir = os.path.join(self.root, 'images')
        segment_dir = os.path.join(self.root, 'segment')
        pos_dir = os.path.join(self.root, 'position')
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
            seg_file = os.path.join(segment_dir, file)
            pos_file = os.path.join(pos_dir, file)
            self.files[self.split].append({
                'img': img_file,
                'seg': seg_file,
                'pos': pos_file,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]

        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        if self.resize:
            img = PIL.Image.Image.resize(img, self._resize_shape)
        img = np.array(img, dtype=np.uint8)

        seg_file = data_file['seg']
        seg = PIL.Image.open(seg_file)
        if self.resize:
            seg = PIL.Image.Image.resize(seg, self._resize_shape)
        seg = np.array(seg, dtype=np.int32)
        seg[seg == 255] = -1

        pos = 0
        if self.pos:
            pos_file = data_file['pos']
            pos = PIL.Image.open(pos_file)
            if self.resize:
                pos = PIL.Image.Image.resize(pos, self._resize_shape)
            pos = np.array(pos, dtype=np.int32)
            pos[pos == 255] = -1

        if self._transform:
            return self.transform(img, seg, pos)
        else:
            return img, seg, pos

    def transform(self, img, seg, pos=0):
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img[np.newaxis, ]
        img = torch.from_numpy(img).float()

        seg = torch.from_numpy(seg).long()
        if self.pos:
            pos = torch.from_numpy(pos).long()
        return img, seg, pos

    def untransform(self, img, seg, pos=0):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img.reshape(img.shape[:-1])

        seg = seg.numpy()
        if self.pos:
            pos = pos.numpy()
        return img, seg, pos


class PhoneSeg(PhoneSegBase):

    def __init__(self, root, split='train', transform=False, resize=False, pos=False):
        self.root = root
        self.split = split
        self._transform = transform
        self._resize_shape = (640, 128)  # (w, h)
        self.resize = resize
        self.pos = pos

        dataset_dir = os.path.join(self.root, 'images')
        segment_dir = os.path.join(self.root, 'segment')
        pos_dir = os.path.join(self.root, 'position')
        namelist_path = os.path.join(self.root, 'namelist.pkl')
        namelist = None
        if os.path.exists(namelist_path):
            with open(namelist_path, 'rb') as fid:
                namelist = cPickle.load(fid)
        train_num = int(len(namelist) * 0.8)

        self.files = collections.defaultdict(list)
        if self.split == 'train':
            namelist = namelist[:train_num]
        elif self.split == 'test':
            namelist = namelist[train_num:]
        
        for file in namelist:
            img_file = os.path.join(dataset_dir, file)
            seg_file = os.path.join(segment_dir, file.split('.')[0]+'.png')
            pos_file = os.path.join(pos_dir, file.split('.')[0]+'.png')
            self.files[self.split].append({
                'img': img_file,
                'seg': seg_file,
                'pos': pos_file,
            })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]

        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        if self.resize:
            img = PIL.Image.Image.resize(img, self._resize_shape)
        img = np.array(img, dtype=np.uint8)

        seg_file = data_file['seg']
        seg = PIL.Image.open(seg_file)
        if self.resize:
            seg = PIL.Image.Image.resize(seg, self._resize_shape)
        seg = np.array(seg, dtype=np.int32)
        seg[seg == 255] = -1

        pos = 0
        if self.pos:
            pos_file = data_file['pos']
            pos = PIL.Image.open(pos_file)
            if self.resize:
                pos = PIL.Image.Image.resize(pos, self._resize_shape)
            pos = np.array(pos, dtype=np.int32)
            pos[pos == 255] = -1

        if self._transform:
            return self.transform(img, seg, pos)
        else:
            return img, seg, pos
