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

from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import *
# import torchvision.datasets as dset

from tensorboardX import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, _data_transforms_cifar10, \
    _data_transforms_cifar100, accuracy, AvgrageMeter, copy_state_dict, copy_optimizer_state_dict
from models.ign_alphas import ign_alphas


def train(args, epoch, train_queue, valid_queue, net, alphas, criterion, optimizer, lr, record_file=None):
    net.train()

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    # offset = 0
    # prev_time = time.time()
    # load_start = time.time()
    for batch_index, ((images, labels, index), (images_valid, labels_valid, index_valid)) in enumerate(
            zip(train_queue, valid_queue)):
        # if epoch <= args.warm:
        #     warmup_scheduler.step()
        net.train()
        n = images.shape[0]
        # indices = [i for i in range(offset, offset + n)]
        # offset += n
        indices = index

        images = Variable(images, requires_grad=False).cuda()
        labels = Variable(labels, requires_grad=False).cuda()
        load_end = time.time()

        # load_val_start=time.time()

        # images_valid, labels_valid = next(iter(valid_queue))
        images_valid = Variable(images_valid, requires_grad=False).cuda()
        labels_valid = Variable(labels_valid, requires_grad=False).cuda()
        print(index)
        break


def infer(args, epoch, valid_queue, net, criterion, mode='val', record_file=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    net.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for step, (images, labels, index) in enumerate(valid_queue):
        images = Variable(images)
        labels = Variable(labels)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        # loss = loss_function(outputs, labels)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        n = images.shape[0]
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
    if mode == 'val':
        print('Valid: epoch {}, loss {:.3f}, top1 {:.3f}, top5 {:.3f}'.format(epoch, objs.avg,
                                                                              top1.avg,
                                                                              top5.avg))
        if record_file is not None:
            with open(record_file, 'a') as f:
                f.write('Valid: epoch {}, loss {:.3f}, top1 {:.3f}, top5 {:.3f}\n'.format(epoch, objs.avg,
                                                                                          top1.avg,
                                                                                          top5.avg))
    else:
        print('Test: epoch {}, loss {:.3f}, top1 {:.3f}, top5 {:.3f}'.format(epoch, objs.avg,
                                                                             top1.avg,
                                                                             top5.avg))
        if record_file is not None:
            with open(record_file, 'a') as f:
                f.write('Test: epoch {}, loss {:.3f}, top1 {:.3f}, top5 {:.3f}\n'.format(epoch, objs.avg,
                                                                                         top1.avg,
                                                                                         top5.avg))

    # print()

    # add informations to tensorboard
    # writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
    # writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='resnet18', help='net type')
    parser.add_argument('--dataset', type=str, default='cifar-10', help='dataset')
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('--workers', type=int, default=2, help='number of workers for dataloader')
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--start_epochs', type=int, default=0)
    parser.add_argument('--batch_size_train', type=int, default=96, help='batch size for dataloader')
    parser.add_argument('--batch_size_val', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('--batch_size_test', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('--print_freq', type=float, default=10, help='report frequency')
    parser.add_argument('--s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=0.025, help='initial learning rate')
    parser.add_argument('--lr_min', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--ignoring', type=bool, default=True, help='whether use ignoring learning')
    parser.add_argument('--train_val_rate', type=float, default=0.75)
    parser.add_argument('--bad_rate', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')

    parser.add_argument('--alpha_epoch', type=int, default=0)
    parser.add_argument('--alphas_lr', type=float, default=1e-3)
    parser.add_argument('--alphas_weight_decay', type=float, default=0)
    parser.add_argument('--nosoftmax', type=bool, default=False)

    parser.add_argument('--save_dir', type=str, default='./result/')
    parser.add_argument('--save_name', type=str, default=None)
    parser.add_argument('--save_epoch', type=int, default=20)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')

    args = parser.parse_args()

    if args.save_name is not None:
        save_path = os.path.join(args.save_dir, args.save_name)
    else:
        directory = args.save_dir
        runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
        run_id = max([int(x.split('_')[-1]) for x in runs]) + 1 if runs else 0
        args.save_name = 'experiment_{}'.format(str(run_id))
        save_path = os.path.join(args.save_dir, args.save_name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    record_file = os.path.join(save_path, 'train_record.txt')

    with open(record_file, 'w') as f:
        f.write('rate:{}, alpha_epoch:{}, epoch:{}\n'.format(args.train_val_rate, args.alpha_epoch, args.epochs))

    if args.ignoring:
        alphas_criterion = nn.CrossEntropyLoss(reduction='none')

    args.softmax = not args.nosoftmax

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    # dataset
    np.random.seed(args.seed)

    if args.dataset == 'cifar-10':
        train_transform, valid_transform = _data_transforms_cifar10(args)
        if args.bad_rate is not None:
            train_data = CIFAR10_bad(root=args.data, train=True, download=True, transform=train_transform,
                                     bad_rate=args.bad_rate)
        else:
            train_data = CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        test_data = CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        train_transform, valid_transform = _data_transforms_cifar100(args)
        if args.bad_rate is not None:
            train_data = CIFAR100_bad(root=args.data, train=True, download=True, transform=train_transform,
                                      bad_rate=args.bad_rate)
        else:
            train_data = CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        test_data = CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_val_rate * num_train))

    # split = 15
    # num_train = 20

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_train,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=args.workers)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size_val,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=args.workers)
    print(len(train_queue), len(valid_queue))

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size_test,
        pin_memory=True, num_workers=args.workers)

    # data preprocessing:
    # cifar100_training_loader, num_train = get_training_dataloader(
    #     settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     num_workers=args.w,
    #     batch_size=args.b,
    #     shuffle=args.s
    # )
    #
    # cifar100_test_loader = get_test_dataloader(
    #     settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     num_workers=args.w,
    #     batch_size=args.b,
    #     shuffle=args.s
    # )

    net = get_network(args, use_gpu=args.gpu, num_train=split)
    net = net.cuda()

    alphas = ign_alphas(net, args)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume is not None:
        checkpoint_path = args.resume
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epochs = checkpoint['epoch'] + 1
        if torch.cuda.device_count() > 1:
            copy_state_dict(net, checkpoint, parallel=True)
        else:
            copy_state_dict(net, checkpoint)
        # copy_optimizer_state_dict(optimizer, checkpoint['optimizer'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # copy_optimizer_state_dict(alphas.optimizer, checkpoint['alpha_optimizer'])
        alphas.optimizer.load_state_dict(checkpoint['alpha_optimizer'])

    # params = net.parameters()
    # for param in params:
    #     # print(param)
    #     print(param.shape)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, last_epoch=args.start_epochs - 1)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    # train_scheduler
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
    #                                                  gamma=0.2)  # learning rate decay
    # iter_per_epoch = len(train_queue)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    best_top1 = 0.0
    for epoch in range(args.start_epochs, args.epochs):
        # if epoch > args.warm:
        #     train_scheduler.step(epoch)
        lr = scheduler.get_lr()[0]

        train(args, epoch, train_queue, valid_queue, net, alphas, criterion,optimizer, lr, record_file)
