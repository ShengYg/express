import argparse
import os

import numpy as np
import skimage
import PIL
import torch
from torch.autograd import Variable
import tqdm

from dataset import PhoneSeg
from network import FCN32s, Front_end, Context
from utils import label_accuracy_score, label_colormap, _fast_hist


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def main(model_name, resize, pos):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    log_dir = os.path.join(os.getcwd(), 'tools_seg', 'logs', 'frontend_TIME-20171120-221751')
    model_file = os.path.join(log_dir, 'phone_640000.h5')
    output_dir = os.path.join(log_dir, 'test')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    root = os.path.join(os.getcwd(), 'data', 'express', 'test_db_benchmark')
    print 'Loading dataset ...'
    test_loader = torch.utils.data.DataLoader(
        PhoneSeg(root, split='all', transform=True, resize=resize, pos=pos),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    n_class = len(test_loader.dataset.class_names)

    print 'Loading net {} ...'.format(model_name)
    model = None
    if model_name == 'fcn32s':
        model = FCN32s(n_class=11, resize = resize)
    elif model_name == 'frontend':
        model = Front_end(n_class=11, pos=pos)
    elif model_name == 'context':
        model = Context(n_class=11, pos=pos)

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))

    load_net(model_file, model)
    model.eval()


    print('==> Evaluating with PhoneSeg')
    hist_seg = np.zeros((n_class, n_class))
    hist_pos = np.zeros((13, 13))
    visualizations = []
    cmap_seg = label_colormap(n_class)
    cmap_seg = (cmap_seg * 255).astype(np.uint8)
    cmap_pos = label_colormap(13)
    cmap_pos = (cmap_pos * 255).astype(np.uint8)
    for batch_idx, (data, target, target_pos) in tqdm.tqdm(enumerate(test_loader),
                                               total=len(test_loader),
                                               ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            if pos:
                target_pos = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        if pos:
            target_pos = Variable(target_pos)
        score, score_pos = model(data)

        imgs = data.data.cpu()
        seg_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        seg_true = target.data.cpu()
        if pos:
            pos_pred = score_pos.data.max(1)[1].cpu().numpy()[:, :, :]
            pos_true = target_pos.data.cpu()
        if pos:
            for img, st, sp, pt, pp in zip(imgs, seg_true, seg_pred, pos_true, pos_pred):
                img, st, pt = test_loader.dataset.untransform(img, st, pt)
                hist_seg += _fast_hist(st.flatten(), sp.flatten(), n_class)
                hist_pos += _fast_hist(pt.flatten(), pp.flatten(), 13)
                if len(visualizations) < 9:
                    imgrgb = skimage.color.gray2rgb(img)
                    st[st == -1] = 0
                    st_viz = cmap_seg[st]
                    sp_viz = cmap_seg[sp]
                    pp_viz = cmap_pos[pp]
                    output = np.vstack((imgrgb, st_viz, sp_viz, pp_viz))
                    PIL.Image.fromarray(output).save(os.path.join(output_dir, '{}.png'.format(batch_idx)), 'png')
                    visualizations.append(batch_idx)
        else:
            for img, st, sp in zip(imgs, seg_true, seg_pred):
                img, st, pt = test_loader.dataset.untransform(img, st)
                hist_seg += _fast_hist(st.flatten(), sp.flatten(), n_class)
                # if len(visualizations) < 9:
                imgrgb = skimage.color.gray2rgb(img)
                st[st == -1] = 0
                st_viz = cmap_seg[st]
                sp_viz = cmap_seg[sp]
                output = np.vstack((imgrgb, st_viz, sp_viz))
                PIL.Image.fromarray(output).save(os.path.join(output_dir, '{}.png'.format(batch_idx)), 'png')
                    # visualizations.append(batch_idx)

    metrics_seg = label_accuracy_score(hist_seg)
    metrics_seg = np.array(metrics_seg)
    metrics_seg *= 100
    print('''\
test segmentation:
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics_seg))
    if pos:
        metrics_pos = label_accuracy_score(hist_pos)
        metrics_pos = np.array(metrics_pos)
        metrics_pos *= 100
        print('''\
test position:
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics_pos))


if __name__ == '__main__':
    # model_name = 'fcn32s'
    # resize = False
    model_name = 'frontend'
    resize = True
    pos = True
    # model_name = 'context'
    # resize = True

    main(model_name, resize, pos)
