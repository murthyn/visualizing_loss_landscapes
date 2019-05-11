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


def get_relative_path(file):
    script_dir = os.path.dirname('')  # <-- absolute dir the script is in
    return os.path.join(script_dir, file)


def load_dataset(dataset='cifar10', datapath='cifar10/data', batch_size=128, \
                 threads=2, raw_data=False, data_split=1, split_idx=0, \
                 trainloader_path="", testloader_path=""):
    """
    Setup dataloader. The data is not randomly cropped as in training because of
    we want to esimate the loss value with a fixed dataset.
    Args:
        raw_data: raw images, no data preprocessing
        data_split: the number of splits for the training dataloader
        split_idx: the index for the split of the dataloader, starting at 0
    Returns:
        train_loader, test_loader
    """

    # use specific dataloaders
    if trainloader_path and testloader_path:
        assert os.path.exists(trainloader_path), 'trainloader does not exist'
        assert os.path.exists(testloader_path), 'testloader does not exist'
        train_loader = torch.load(trainloader_path)
        test_loader = torch.load(testloader_path)
        return train_loader, test_loader

    assert split_idx < data_split, 'the index of data partition should be smaller than the total number of split'

    if dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])

        data_folder = get_relative_path(datapath)
        if raw_data:
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

        trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                                download=True, transform=transform)
        # If data_split>1, then randomly select a subset of the data. E.g., if datasplit=3, then
        # randomly choose 1/3 of the data.
        if data_split > 1:
            indices = torch.tensor(np.arange(len(trainset)))
            data_num = len(trainset) // data_split # the number of data in a chunk of the split

            # Randomly sample indices. Use seed=0 in the generator to make this reproducible
            state = np.random.get_state()
            np.random.seed(0)
            indices = np.random.choice(indices, data_num, replace=False)
            np.random.set_state(state)

            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                       sampler=train_sampler,
                                                       shuffle=False, num_workers=threads)
        else:
            kwargs = {'num_workers': 2, 'pin_memory': True}
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                      shuffle=False, **kwargs)
        testset = torchvision.datasets.CIFAR10(root=data_folder, train=False,
                                               download=False, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, num_workers=threads)

    return train_loader, test_loader
