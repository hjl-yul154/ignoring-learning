# train.py
# !/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import *
# import torchvision.datasets as dset
from torchvision.transforms import ToPILImage
from IPython.display import Image

from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, _data_transforms_cifar10, \
    accuracy, AvgrageMeter, copy_state_dict, copy_optimizer_state_dict
from models.ign_alphas import ign_alphas

img_class=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='resnet18', help='net type')
    parser.add_argument('--dataset', type=str, default='cifar-10', help='dataset')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--w', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--batch_size_train', type=int, default=96, help='batch size for dataloader')
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('--batch_size_test', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('--print_freq', type=float, default=20, help='report frequency')
    parser.add_argument('--s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--lr_min', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--ignoring', type=bool, default=True, help='whether use ignoring learning')
    parser.add_argument('--train_val_rate', type=float, default=0.75)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    parser.add_argument('--alpha_epoch', type=int, default=20)
    parser.add_argument('--alphas_lr', type=float, default=1e-4)
    parser.add_argument('--alphas_weight_decay', type=float, default=0)

    parser.add_argument('--save_dir', type=str, default='./result/')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--save_epoch', type=int, default=20)
    parser.add_argument('--resume', type=str, default='./result/experiment_3/best_model.pth')

    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')

    args = parser.parse_args()

    # if args.save_name is not None:
    #     save_path = os.path.join(args.save_dir, args.save_name)
    # else:
    #     directory = args.save_dir
    #     runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
    #     run_id = max([int(x.split('_')[-1]) for x in runs]) + 1 if runs else 0
    #     args.save_name = 'experiment_{}'.format(str(run_id))
    #     save_path = os.path.join(args.save_dir, args.save_name)
    # if not os.path.isdir(save_path):
    #     os.mkdir(save_path)
    #
    # if args.ignoring:
    #     alphas_criterion = nn.CrossEntropyLoss(reduction='none')

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # dataset
    np.random.seed(args.seed)
    train_transform, valid_transform = _data_transforms_cifar10(args)
    show_transform=transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = CIFAR10(root=args.data, train=True, download=True, transform=show_transform)
    test_data = CIFAR10(root=args.data, train=False, download=True, transform=show_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_val_rate * num_train))

    # split = 15
    # num_train = 20

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_train,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_val,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)
    print(len(train_queue), len(valid_queue))

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size_test,
        pin_memory=True, num_workers=2)
    if args.resume is None:
        raise RuntimeError("no resume checkpoint")
    checkpoint_path = args.resume
    if not os.path.isfile(checkpoint_path):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

    checkpoint = torch.load(checkpoint_path)
    alphas = checkpoint['alpha_state_dict']
    sorted_index = torch.argsort(alphas)[0:10]
    for i in sorted_index:
        value = alphas[i]
        image, labels, index = train_data[i]
        image=(image-torch.min(image))/(torch.max(image)-torch.min(image))
        image = np.transpose(image.numpy(), (1, 2, 0))
        plt.imshow(image)
        plt.title('value: {}, class {}'.format(value,img_class[labels]))
        plt.show()
