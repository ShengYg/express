import argparse
import os

import numpy as np
import skimage
import PIL
import torch
from torch.autograd import Variable
import tqdm

from dataset import PhoneSeg
from fcn32s import FCN32s
from utils import label_accuracy_score, label_colormap


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    log_dir = os.path.join(os.getcwd(), 'tools_seg', 'logs', 'fcn32s_TIME-20171031-133446')
    model_file = os.path.join(log_dir, 'phone_400000.h5')
    output_dir = os.path.join(log_dir, 'test')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    root = os.path.join(os.getcwd(), 'data', 'express', 'pretrain_db_benchmark')
    test_loader = torch.utils.data.DataLoader(
        PhoneSeg(root, split='test', transform=True),
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True)

    n_class = len(test_loader.dataset.class_names)

    model = FCN32s(n_class=11)
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))

    load_net(model_file, model)
    model.eval()


    print('==> Evaluating with PhoneSeg')
    visualizations = []
    label_trues, label_preds = [], []
    cmap = label_colormap(n_class)
    cmap = (cmap * 255).astype(np.uint8)
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(test_loader),
                                               total=len(test_loader),
                                               ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        score = model(data)

        imgs = data.data.cpu()
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu()
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            img, lt = test_loader.dataset.untransform(img, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                imgrgb = skimage.color.gray2rgb(img)
                lt[lt == -1] = 0
                lt_viz = cmap[lt]
                lp_viz = cmap[lp]
                output = np.vstack((imgrgb, lt_viz, lp_viz))
                PIL.Image.fromarray(output).save(os.path.join(output_dir, '{}.png'.format(batch_idx)), 'png')
                visualizations.append(batch_idx)
    metrics = label_accuracy_score(
        label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
Accuracy: {0}
Accuracy Class: {1}
Mean IU: {2}
FWAV Accuracy: {3}'''.format(*metrics))


if __name__ == '__main__':
    main()
