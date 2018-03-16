import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from utils.blob import im_list_to_blob
from fast_rcnn.config import cfg
import network
from network import Conv2d, FC

class VGG16_PHONE(nn.Module):
    def __init__(self, bn=False):
        super(VGG16_PHONE, self).__init__()

        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(Conv2d(64, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   Conv2d(128, 128, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))

        self.conv3 = nn.Sequential(Conv2d(128, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   Conv2d(256, 256, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(Conv2d(256, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   nn.MaxPool2d(2))
        self.conv5 = nn.Sequential(Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn),
                                   Conv2d(512, 512, 3, same_padding=True, bn=bn))

    def forward(self, im_data):

        x = self.conv1(im_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

class PhoneNet(nn.Module):
    n_classes = 11
    classes = np.asarray([ 'zero', 'one', 'two', 'three',
                           'four', 'five', 'six', 'seven', 'eight',
                           'nine', '__background__'])

    def __init__(self, classes=None, debug=False, bn=False):
        super(PhoneNet, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
        self.bn = bn
        self.features = VGG16_PHONE(bn=bn)

        self.fc6 = FC(512 * 16 * 3, 1024)
        
        # self.fc_pool1 = nn.MaxPool2d((1,2))
        # self.fc_pool2 = nn.MaxPool2d((1,4))
        # self.fc_pool3 = nn.MaxPool2d((1,8))
        # self.fc_pool4 = nn.MaxPool2d((1,16))
        # self.fc6 = FC(512 * 31 * 3, 1024)

        self.score_fc = nn.ModuleList([FC(1024, self.n_classes) for i in range(12)])
        self.length_fc = FC(1024, 8)

        self.length = 13
        self.out_loss = [None] * 13
        self.cls_score = [None] * 13
        self.cls_prob = [None] * 13

    @property
    def loss(self):
        return self.out_loss[0] + self.out_loss[1] + self.out_loss[2] + self.out_loss[3] + self.out_loss[4] + self.out_loss[5] \
        + self.out_loss[6] + self.out_loss[7] + self.out_loss[8] + self.out_loss[9] + self.out_loss[10] + self.out_loss[11] \
        + self.out_loss[12]

    @property
    def loss_5k(self):
        return self.out_loss[1] + self.out_loss[2] + self.out_loss[3] + self.out_loss[4] + self.out_loss[5] \
        + self.out_loss[6] + self.out_loss[7] + self.out_loss[8] + self.out_loss[9] + self.out_loss[10] + self.out_loss[11] \
        + self.out_loss[12]

    @property
    def loss_10k(self):
        return self.out_loss[3] + self.out_loss[4] + self.out_loss[5] \
        + self.out_loss[6] + self.out_loss[7] + self.out_loss[8] + self.out_loss[9] + self.out_loss[10] + self.out_loss[11]

    def forward(self, im_data, labels=None, length=None):
        im_data = network.np_to_variable(im_data, is_cuda=True)
        features = self.features(im_data)

        x = features.view(features.size()[0], -1)
        y = self.fc6(x)
        if not self.bn:
            y = F.dropout(y, training=self.training)

        
        # y = torch.cat([features, self.fc_pool1(features), self.fc_pool2(features), self.fc_pool3(features), self.fc_pool4(features)], dim=-1)
        # y = y.view(y.size()[0], -1)
        # y = self.fc6(y)
        # if not self.bn:
        #     y = F.dropout(y, training=self.training)

        ignore_weights = torch.from_numpy(np.ones(self.n_classes, dtype=np.float32))
        ignore_weights[10] = 0

        for i in range(12):
            self.cls_score[i] = self.score_fc[i](y)
            self.cls_score[i] = self.cls_score[i].view(self.cls_score[i].size()[0], -1)
            if self.training:
                self.out_loss[i] = F.cross_entropy(self.cls_score[i], network.np_to_variable(labels[:, i], is_cuda=True, dtype=torch.LongTensor), weight=ignore_weights.cuda())
            self.cls_prob[i] = F.softmax(self.cls_score[i], dim=1)
        
        # cls length
        self.cls_score[12] = self.length_fc(y)
        self.cls_score[12] = self.cls_score[12].view(self.cls_score[12].size()[0], -1)
        if self.training:
            self.out_loss[12] = F.cross_entropy(self.cls_score[12], network.np_to_variable(length, is_cuda=True, dtype=torch.LongTensor))
        self.cls_prob[12] = F.softmax(self.cls_score[12], dim=1)

        return self.cls_prob


    def get_image_blob(self, im, height=48, width=256):
        
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        processed_ims = []

        ## v1
        im_scale_x = float(width) / float(im_shape[1])
        im_scale_y = float(height) / float(im_shape[0])
        im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        ## v2
        # im_scale = float(width) / float(im_shape[1])
        # if im_scale * im_shape[0] > height:
        #     im_scale = float(height) / float(im_shape[0])
        # im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
        #                 interpolation=cv2.INTER_LINEAR)
        processed_ims.append(im)

        # Create a blob to hold the input images
        blob = self._im_list_to_blob(processed_ims, height=height, width=width)

        return blob

    def get_image_blob_list(self, im_list, height=48, width=256):

        processed_ims = []
        for im in im_list:
            im_orig = im.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS

            im_shape = im_orig.shape

            ## v1
            im_scale_x = float(width) / float(im_shape[1])
            im_scale_y = float(height) / float(im_shape[0])
            im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                            interpolation=cv2.INTER_LINEAR)
            ## v2
            # im_scale = float(width) / float(im_shape[1])
            # if im_scale * im_shape[0] > height:
            #     im_scale = float(height) / float(im_shape[0])
            # im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            #                 interpolation=cv2.INTER_LINEAR)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = self._im_list_to_blob(processed_ims, height=height, width=width)

        return blob


    def _im_list_to_blob(self, ims, height=48, width=256):
        num_images = len(ims)
        blob = np.zeros((num_images, height, width, 3), dtype=np.float32)
        for i in xrange(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
            # blob[i, height/2-im.shape[0]/2:height/2+(im.shape[0]+1)/2, 
            #         width/2-im.shape[1]/2:width/2+(im.shape[1]+1)/2, :] = im
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob


