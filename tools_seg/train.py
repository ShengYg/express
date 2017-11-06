import datetime
import os
import torch
import yaml
import pprint

from dataset import PhoneSeg
from network import FCN32s, Front_end, Context, load_pretrain_net
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

def main(model_name, resize):
    cfg = dict(
        max_iteration=2000000,
        snap_interval=10000,
        log_interval=100,
        lr=1.0e-7,
        lr_decay_interval=[400000,1000000],
        momentum=0.99,
        weight_decay=0.0005,
    )
    print '## cfg:'
    pprint.pprint(cfg)
    out = get_log_dir(model_name, cfg)

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
        PhoneSeg(root, split='train', transform=True, resize=resize),
        batch_size=1, shuffle=True, **kwargs)

    # 2. model
    print '## Loading model'
    model = None
    if model_name == 'fcn32s':
        model = FCN32s(n_class=11, resize = resize)
    elif model_name == 'frontend':
        model = Front_end(n_class=11)
        weight_path = os.path.join(os.getcwd(), 'output', 'mnist_out', 'mnist_14800.h5')
        if os.path.exists(weight_path):
            load_pretrain_net(weight_path, model, num=16)
    elif model_name == 'context':
        model = Context(n_class=11)
        weight_path = os.path.join(os.getcwd(), 'tools_seg', 'phone_450000.h5')
        if os.path.exists(weight_path):
            load_pretrain_net(weight_path, model, num=28)
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

    # model_name = 'fcn32s'
    # model_name = 'frontend'
    model_name = 'context'
    resize = True
    main(model_name, resize)