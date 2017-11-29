import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                      dtype=np.float64)
    weight[range(in_channels), range(out_channels), :, :] = filt
    return torch.from_numpy(weight).float()

def load_pretrain_net(fname, net, num=8*2):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items()[:num]:
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)

class FCN32s(nn.Module):

    def __init__(self, n_class=11, resize=False):
        super(FCN32s, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=10)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.conv3_5 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu3_5 = nn.ReLU(inplace=True)
        self.conv3_6 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu3_6 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # fc6
        self.fc6 = nn.Conv2d(512, 1024, 4)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        # fc7
        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(1024, n_class, 1)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 16, stride=8,
                                          bias=False)

        self._initialize_weights()
        self.resize = resize


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)

    def forward(self, x):
        h = x

        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.relu2_3(self.conv2_3(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.relu3_4(self.conv3_4(h))
        h = self.relu3_5(self.conv3_5(h))
        h = self.relu3_6(self.conv3_6(h))
        h = self.pool3(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)

        h = self.upscore(h)
        if self.resize:
            h = h[:, :, 5:5 + x.size()[2], 5:5 + x.size()[3]].contiguous()  # for input 96*480
        else:
            h = h[:, :, 1:1 + x.size()[2], 1:1 + x.size()[3]].contiguous()  # for any size

        return h

class Front_end(nn.Module):

    def __init__(self, n_class=11, pos=False):
        super(Front_end, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=2, dilation=2)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.relu4_3 = nn.ReLU(inplace=True)

        self.fc6 = nn.Conv2d(512, 1024, 3, padding=4, dilation=4)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        self.final = nn.Conv2d(1024, n_class, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

        self.pos = pos
        if self.pos:
            self.fc6_pos = nn.Conv2d(512, 1024, 3, padding=4, dilation=4)
            self.relu6_pos = nn.ReLU(inplace=True)
            self.drop6_pos = nn.Dropout2d()
            self.fc7_pos = nn.Conv2d(1024, 1024, 1)
            self.relu7_pos = nn.ReLU(inplace=True)
            self.drop7_pos = nn.Dropout2d()
            self.final_pos = nn.Conv2d(1024, 13, 1)
            self.upsample_pos = nn.Upsample(scale_factor=4, mode='bilinear')

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu1_1(self.conv1_1(x))
        x = self.relu1_2(self.conv1_2(x))
        x = self.pool1(x)

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.relu2_3(self.conv2_3(x))
        x = self.pool2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))

        h1 = self.relu6(self.fc6(x))
        h1 = self.drop6(h1)
        h1 = self.relu7(self.fc7(h1))
        h1 = self.drop7(h1)
        h1 = self.final(h1)
        h1 = self.upsample(h1)

        h2 = None
        if self.pos:
            h2 = self.relu6_pos(self.fc6_pos(x))
            h2 = self.drop6_pos(h2)
            h2 = self.relu7_pos(self.fc7_pos(h2))
            h2 = self.drop7_pos(h2)
            h2 = self.final_pos(h2)
            h2 = self.upsample_pos(h2)

        return (h1, h2)

class Context(nn.Module):

    def __init__(self, n_class=11):
        super(Context, self).__init__()
        # conv1
        self.conv1_1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.conv2_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=2, dilation=2)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.relu4_3 = nn.ReLU(inplace=True)

        self.fc6 = nn.Conv2d(512, 1024, 3, padding=4, dilation=4)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.final = nn.Conv2d(1024, n_class, 1)

        self.ct_conv1_1  = nn.Conv2d(n_class, n_class, 3, padding=1, dilation=1)
        self.ct_relu1_1  = nn.ReLU(inplace=True)
        self.ct_conv1_2  = nn.Conv2d(n_class, n_class, 3, padding=1, dilation=1)
        self.ct_relu1_2  = nn.ReLU(inplace=True)
        self.ct_conv2_1  = nn.Conv2d(n_class, n_class, 3, padding=2, dilation=2)
        self.ct_relu2_1  = nn.ReLU(inplace=True)
        self.ct_conv3_1  = nn.Conv2d(n_class, n_class, 3, padding=4, dilation=4)
        self.ct_relu3_1  = nn.ReLU(inplace=True)
        self.ct_conv4_1  = nn.Conv2d(n_class, n_class, 3, padding=8, dilation=8)
        self.ct_relu4_1  = nn.ReLU(inplace=True)
        self.ct_fc1      = nn.Conv2d(n_class, n_class, 3, padding=1, dilation=1)
        self.ct_fc1_relu = nn.ReLU(inplace=True)
        self.ct_final = nn.Conv2d(n_class, n_class, 1)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear')

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.weight.data.shape[2] == 3:
                    for i in range(min(m.weight.data.shape[:2])):
                        m.weight.data[i][i][1][1] = 1
                elif m.weight.data.shape[2] == 1:
                    for i in range(min(m.weight.data.shape[:2])):
                        m.weight.data[i][i][0][0] = 1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        h = x

        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.relu2_3(self.conv2_3(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.final(h)

        h = self.ct_relu1_1(self.ct_conv1_1(h))
        h = self.ct_relu1_2(self.ct_conv1_2(h))
        h = self.ct_relu2_1(self.ct_conv2_1(h))
        h = self.ct_relu3_1(self.ct_conv3_1(h))
        h = self.ct_relu4_1(self.ct_conv4_1(h))
        h = self.ct_fc1_relu(self.ct_fc1(h))
        h = self.ct_final(h)

        h = self.upsample(h)

        return h
