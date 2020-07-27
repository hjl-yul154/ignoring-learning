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
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
# from dataset import *
from torch.autograd import Variable
import torchvision.datasets as dset

from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, _data_transforms_cifar10, \
    accuracy, AvgrageMeter
from models.ign_alphas import ign_alphas


def train(args, epoch, train_queue, valid_queue, net, alphas, criterion, optimizer, lr):
    net.train()

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    offset = 0
    prev_time = time.time()
    load_start = time.time()
    for batch_index, ((images, labels), (images_valid, labels_valid)) in enumerate(zip(train_queue, valid_queue)):

        # if epoch <= args.warm:
        #     warmup_scheduler.step()
        net.train()
        n = images.shape[0]
        indices = [i for i in range(offset, offset + n)]
        offset += n

        images = Variable(images, requires_grad=False).cuda()
        labels = Variable(labels, requires_grad=False).cuda()
        load_end = time.time()


    print('epoch {}, loss: {}, top1: {}, top5: {}'.format(epoch, objs.avg,
                                                          top1.avg,
                                                          top5.avg))
    return top1.avg, objs.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='resnet18', help='net type')
    parser.add_argument('--dataset', type=str, default='cifar-10', help='dataset')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--w', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size_train', type=int, default=3, help='batch size for dataloader')
    parser.add_argument('--batch_size_val', type=int, default=1, help='batch size for dataloader')
    parser.add_argument('--print_freq', type=float, default=5, help='report frequency')
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

    parser.add_argument('--alphas_lr', type=float, default=1e-4)
    parser.add_argument('--alphas_weight_decay', type=float, default=0)
    # parser.add_argument()

    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')

    args = parser.parse_args()

    if args.ignoring:
        alphas_criterion = nn.CrossEntropyLoss(reduction='none')

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # dataset
    np.random.seed(args.seed)
    train_transform, valid_transform = _data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_val_rate * num_train))

    split=15
    num_train=20

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_train,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_val,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=2)
    print(len(train_queue), len(valid_queue))


    net = get_network(args, use_gpu=args.gpu, num_train=split)
    net = net.cuda()

    # params = net.parameters()
    # for param in params:
    #     # print(param)
    #     print(param.shape)

    alphas = ign_alphas(net, args)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.lr_min,last_epoch=5)

    print(scheduler.last_epoch)
    scheduler.step()
    print(scheduler.last_epoch)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # train_scheduler
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
    #                                                  gamma=0.2)  # learning rate decay
    # iter_per_epoch = len(train_queue)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    # checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)
    #
    # # use tensorboard
    # if not os.path.exists(settings.LOG_DIR):
    #     os.mkdir(settings.LOG_DIR)
    #
    # # create checkpoint folder to save model
    # if not os.path.exists(checkpoint_path):
    #     os.makedirs(checkpoint_path)
    # checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    for epoch in range(args.epochs):
        # if epoch > args.warm:
        #     train_scheduler.step(epoch)
        lr = scheduler.get_lr()[0]

        train(args, epoch, train_queue, valid_queue, net, alphas, criterion, optimizer, lr)
        scheduler.step()

        # acc = infer(args, epoch, valid_queue, net, criterion)

        # # start to save best performance model after learning rate decay to 0.01
        # if epoch > settings.MILESTONES[1] and best_acc < acc:
        #     torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
        #     best_acc = acc
        #     continue
        #
        # if not epoch % settings.SAVE_EPOCH:
        #     torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    # writer.close()
