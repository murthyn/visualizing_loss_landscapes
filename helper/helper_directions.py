# CREATE DIRECTIONS: Supporting functions (derived from https://github.com/tomgoldstein/loss-landscape/blob/master/net_plotter.py)

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

from helper_weights_states import get_weights, set_weights, set_states, get_random_weights, get_random_states, get_diff_weights, get_diff_states
from helper_normalize import normalize_direction, normalize_directions_for_weights, normalize_directions_for_states, ignore_biasbn
from helper_h5_util import write_list, read_list

def create_target_direction(net, net2, dir_type='states'):
    """
        Setup a target direction from one model to the other
        Args:
          net: the source model
          net2: the target model with the same architecture as net.
          dir_type: 'weights' or 'states', type of directions.
        Returns:
          direction: the target direction from net to net2 with the same dimension
                     as weights or states.
    """

    assert (net2 is not None)
    # direction between net2 and net
    if dir_type == 'weights':
        w = get_weights(net)
        w2 = get_weights(net2)
        direction = get_diff_weights(w, w2)
    elif dir_type == 'states':
        s = net.state_dict()
        s2 = net2.state_dict()
        direction = get_diff_states(s, s2)

    return direction
  
def create_random_direction(net, dir_type='weights', ignore='biasbn', norm='filter'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.
        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'
        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    if dir_type == 'weights':
        weights = get_weights(net) # a list of parameters.
        direction = get_random_weights(weights)
        normalize_directions_for_weights(direction, weights, norm, ignore)
    elif dir_type == 'states':
        states = net.state_dict() # a dict of parameters, including BN's running mean/var.
        direction = get_random_states(states)
        normalize_directions_for_states(direction, states, norm, ignore)

    return direction
  
def setup_direction(args, dir_file, net):
    """
        Setup the h5 file to store the directions.
        - xdirection, ydirection: The pertubation direction added to the mdoel.
          The direction is a list of tensors.
    """
    print('-------------------------------------------------------------------')
    print('setup_direction')
    print('-------------------------------------------------------------------')
    # Skip if the direction file already exists
    if exists(dir_file):
        f = h5py.File(dir_file, 'r')
        if (args.y and 'ydirection' in f.keys()) or 'xdirection' in f.keys():
            f.close()
            print ("%s is already setted up" % dir_file)
            return
        f.close()

    # Create the plotting directions
    f = h5py.File(dir_file,'w') # create file, fail if exists
    if not args.dir_file:
        print("Setting up the plotting directions...")
        if args.model_file2:
            net2 = model_loader.load(args.dataset, args.model, args.model_file2)
            xdirection = create_target_direction(net, net2, args.dir_type)
        else:
            xdirection = create_random_direction(net, args.dir_type, args.xignore, args.xnorm)
        write_list(f, 'xdirection', xdirection)

        if args.y:
            if args.same_dir:
                ydirection = xdirection
            elif args.model_file3:
                net3 = model_loader.load(args.dataset, args.model, args.model_file3)
                ydirection = create_target_direction(net, net3, args.dir_type)
            else:
                ydirection = create_random_direction(net, args.dir_type, args.yignore, args.ynorm)
            write_list(f, 'ydirection', ydirection)

    f.close()
    print ("direction file created: %s" % dir_file)
    
def name_direction_file(args):
    """ Name the direction file that stores the random directions. """

    if args.dir_file:
        assert exists(args.dir_file), "%s does not exist!" % args.dir_file
        return args.dir_file

    dir_file = ""

    file1, file2, file3 = args.model_file, args.model_file2, args.model_file3

    # name for xdirection
    if file2:
        # 1D linear interpolation between two models
        assert exists(file2), file2 + " does not exist!"
        if file1[:file1.rfind('/')] == file2[:file2.rfind('/')]:
            # model_file and model_file2 are under the same folder
            dir_file += file1 + '_' + file2[file2.rfind('/')+1:]
        else:
            # model_file and model_file2 are under different folders
            prefix = commonprefix([file1, file2])
            prefix = prefix[0:prefix.rfind('/')]
            dir_file += file1[:file1.rfind('/')] + '_' + file1[file1.rfind('/')+1:] + '_' + \
                       file2[len(prefix)+1: file2.rfind('/')] + '_' + file2[file2.rfind('/')+1:]
    else:
        dir_file += file1

    dir_file += '_' + args.dir_type
    if args.xignore:
        dir_file += '_xignore=' + args.xignore
    if args.xnorm:
        dir_file += '_xnorm=' + args.xnorm

    # name for ydirection
    if args.y:
        if file3:
            assert exists(file3), "%s does not exist!" % file3
            if file1[:file1.rfind('/')] == file3[:file3.rfind('/')]:
               dir_file += file3
            else:
               # model_file and model_file3 are under different folders
               dir_file += file3[:file3.rfind('/')] + '_' + file3[file3.rfind('/')+1:]
        else:
            if args.yignore:
                dir_file += '_yignore=' + args.yignore
            if args.ynorm:
                dir_file += '_ynorm=' + args.ynorm
            if args.same_dir: # ydirection is the same as xdirection
                dir_file += '_same_dir'

    # index number
    if args.idx > 0: dir_file += '_idx=' + str(args.idx)

    dir_file += ".h5"

    return dir_file
  
def load_directions(dir_file):
    """ Load direction(s) from the direction file."""

    f = h5py.File(dir_file, 'r')
    if 'ydirection' in f.keys():  # If this is a 2D plot
        xdirection = read_list(f, 'xdirection')
        ydirection = read_list(f, 'ydirection')
        directions = [xdirection, ydirection]
    else:
        directions = [read_list(f, 'xdirection')]

    return directions