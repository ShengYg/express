# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.express import express
from datasets.phone import phone
from datasets.mnist import mnist
import numpy as np


# Set up express_<split>
for split in ['train', 'test']:
    name = 'express_{}'.format(split)
    __sets[name] = (lambda split=split, root_dir=None, ratio=0.8: express(split, root_dir, ratio))

# Set up phone_<split>
for split in ['train', 'test']:
    name = 'phone_{}'.format(split)
    __sets[name] = (lambda split=split, root_dir=None, ratio=0.8: phone(split, root_dir, ratio))

# Set up mnist_<split>
for split in ['train', 'test']:
    name = 'mnist_{}'.format(split)
    __sets[name] = (lambda split=split, root_dir=None, ratio=0.8: mnist(split, root_dir, ratio))

def get_imdb(name, root_dir = None, ratio=0.8):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name](root_dir=root_dir, ratio=ratio)

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
