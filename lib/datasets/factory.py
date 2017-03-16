# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.psdb import psdb
from datasets.psdb import psdbcrop
from datasets.express import express
import numpy as np


# Set up psdb_<split>
for split in ['train', 'test']:
    name = 'psdb_{}'.format(split)
    __sets[name] = (lambda split=split: psdb(split))

# Set up psdbcrop_<split>
for split in ['train', 'test']:
    name = 'psdbcrop_{}'.format(split)
    __sets[name] = (lambda split=split: psdbcrop(split))

# Set up express_<split>
for split in ['train', 'test']:
    name = 'express_{}'.format(split)
    __sets[name] = (lambda split=split: express(split))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
