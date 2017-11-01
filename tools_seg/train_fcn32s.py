import datetime
import os
import torch
import yaml
import pprint

from dataset import PhoneSeg
from fcn32s import FCN32s
from trainer import Trainer



def get_log_dir(model_name, cfg):
    # load config
    name = model_name
    now = datetime.datetime.now()
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    # create out
    log_dir = os.path.join(here, 'logs', name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


here = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg = dict(
        max_iteration=100000,
        snap_interval=1000,
        log_interval=100,
        lr=1.0e-6,
        lr_decay_interval=[20000,50000],
        momentum=0.99,
        weight_decay=0.0005,
    )
    print '## cfg:'
    pprint.pprint(cfg)
    out = get_log_dir('fcn32s', cfg)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    cuda = torch.cuda.is_available()

    torch.manual_seed(2017)
    if cuda:
        torch.cuda.manual_seed(2017)

    # 1. dataset
    print '## Loading dataset'
    root = os.path.join(os.getcwd(), 'data', 'express', 'pretrain_db_benchmark')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(
        PhoneSeg(root, split='train', transform=True, resize=True),
        batch_size=20, shuffle=True, **kwargs)

    # 2. model
    print '## Loading model'
    model = FCN32s(n_class=11)
    start_epoch = 0
    start_iteration = 0
    if cuda:
        model = model.cuda()

    trainer = Trainer(
        cuda=cuda,
        model=model,
        train_loader=train_loader,
        out=out,
        size_average=True,
        cfg=cfg,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
