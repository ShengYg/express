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

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Sequential(Conv2d(3, 64, 3, same_padding=True),
                                   Conv2d(64, 64, 3, same_padding=True),
                                   nn.MaxPool2d(2))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool2d((15, 3))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.avgpool(x)
        return x
        

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

    def __init__(self, classes=None, debug=False):
        super(PhoneNet, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
        self.features = VGG16_PHONE(bn=False)
        # self.features = ResNet(Bottleneck, [2,3,5,2])

        # self.conv5 = nn.ModuleList([nn.Sequential(Conv2d(512, 1024, 3, same_padding=True),
        #                    Conv2d(1024, 1024, 3, same_padding=True),
        #                    Conv2d(1024, self.n_classes, 3, same_padding=True)) for i in range(12)] + 
        #                    [nn.Sequential(Conv2d(512, 1024, 3, same_padding=True),
        #                    Conv2d(1024, 1024, 3, same_padding=True),
        #                    Conv2d(1024, 8, 3, same_padding=True))])
        # self.score_fc = nn.ModuleList([nn.AvgPool2d((3, 15)) for i in range(12)])
        # self.length_fc = nn.AvgPool2d((3, 15))
        # self.fc6 = nn.ModuleList([FC(512 * 15 * 3, 1024) for i in range(13)])
        self.fc6 = FC(512 * 15 * 3, 1024)
        # self.fc6 = FC(2048 * 15 * 3, 1024)
        # self.fc6 = nn.AvgPool2d((3, 15))
        self.score_fc = nn.ModuleList([FC(1024, self.n_classes) for i in range(12)])
        self.length_fc = FC(1024, 8)

        # loss
        self.length = 13
        self.out_loss = [None] * 13
        self.cls_score = [None] * 13
        self.cls_prob = [None] * 13

        # for log
        self.debug = debug

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
        # y = y.view(y.size()[0], -1)
        y = F.dropout(y, training=self.training)

        ignore_weights = torch.from_numpy(np.ones(self.n_classes, dtype=np.float32))
        ignore_weights[10] = 0

        for i in range(12):
            # y = self.conv5[i](features)
            self.cls_score[i] = self.score_fc[i](y)
            self.cls_score[i] = self.cls_score[i].view(self.cls_score[i].size()[0], -1)
            if self.training:
                self.out_loss[i] = F.cross_entropy(self.cls_score[i], network.np_to_variable(labels[:, i], is_cuda=True, dtype=torch.LongTensor), weight=ignore_weights.cuda())
            self.cls_prob[i] = F.softmax(self.cls_score[i])
        
        # cls length
        # y = self.conv5[12](features)
        self.cls_score[12] = self.length_fc(y)
        self.cls_score[12] = self.cls_score[12].view(self.cls_score[12].size()[0], -1)
        if self.training:
            self.out_loss[12] = F.cross_entropy(self.cls_score[12], network.np_to_variable(length, is_cuda=True, dtype=torch.LongTensor))
        self.cls_prob[12] = F.softmax(self.cls_score[12])

        return self.cls_prob


    def get_image_blob(self, im):
        
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= cfg.PIXEL_MEANS

        im_shape = im_orig.shape
        im_scale_x = float(cfg.TEST.WIDTH) / float(im_shape[1])
        im_scale_y = float(cfg.TEST.HEIGHT) / float(im_shape[0])

        processed_ims = []
        im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                        interpolation=cv2.INTER_LINEAR)
        processed_ims.append(im)

        # Create a blob to hold the input images
        blob = self._im_list_to_blob(processed_ims)

        return blob

    def get_image_blob_list(self, im_list):

        processed_ims = []
        for im in im_list:
            im_orig = im.astype(np.float32, copy=True)
            im_orig -= cfg.PIXEL_MEANS

            im_shape = im_orig.shape
            # print im_shape
            im_scale_x = float(cfg.TEST.WIDTH) / float(im_shape[1])
            im_scale_y = float(cfg.TEST.HEIGHT) / float(im_shape[0])

            im = cv2.resize(im_orig, None, None, fx=im_scale_x, fy=im_scale_y,
                            interpolation=cv2.INTER_LINEAR)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = self._im_list_to_blob(processed_ims)

        return blob


    def _im_list_to_blob(self, ims):
        img_shape = ims[0].shape   
        num_images = len(ims)
        blob = np.zeros((num_images, img_shape[0], img_shape[1], 3),    
                        dtype=np.float32)           #[nums, h, w, 3]
        for i in xrange(num_images):
            im = ims[i]
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        # Axis order will become: (batch elem, channel, height, width)
        channel_swap = (0, 3, 1, 2)
        blob = blob.transpose(channel_swap)
        return blob

    # def load_from_npz(self, params):
    #     self.rpn.load_from_npz(params)

    #     pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7', 'score_fc.fc': 'cls_score', 'bbox_fc.fc': 'bbox_pred'}
    #     own_dict = self.state_dict()
    #     for k, v in pairs.items():
    #         key = '{}.weight'.format(k)
    #         param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(1, 0)
    #         own_dict[key].copy_(param)

    #         key = '{}.bias'.format(k)
    #         param = torch.from_numpy(params['{}/biases:0'.format(v)])
    #         own_dict[key].copy_(param)

