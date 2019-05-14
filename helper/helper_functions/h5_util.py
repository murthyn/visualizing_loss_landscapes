# H5 UTIL: Supporting functions (derived from https://github.com/tomgoldstein/loss-landscape/blob/master/h5_util.py)

import torch, torchvision
from torchvision import transforms
from torch.autograd.variable import Variable

import os, copy
from os.path import exists, commonprefix

import h5py
import numpy as np
from matplotlib import pyplot as pp

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

def write_list(f, name, direction):
    """ Save the direction to the hdf5 file with name as the key
        Args:
            f: h5py file object
            name: key name_surface_file
            direction: a list of tensors
    """

    grp = f.create_group(name)
    for i, l in enumerate(direction):
        if isinstance(l, torch.Tensor):
            l = l.numpy()
        grp.create_dataset(str(i), data=l)


def read_list(f, name):
    """ Read group with name as the key from the hdf5 file and return a list numpy vectors. """
    grp = f[name]
    return [grp[str(i)] for i in range(len(grp))]