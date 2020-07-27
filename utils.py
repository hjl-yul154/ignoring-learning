""" helper function

author baiyu
"""

import sys

import numpy as np

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn


# from dataset import CIFAR100Train, CIFAR100Test

def get_network(args, use_gpu=True, num_train=0):
    """ return given network
    """
    if args.dataset == 'cifar-10':
        num_classes = 10
    elif args.dataset == 'cifar-100':
        num_classes = 100
    else:
        num_classes = 0

    if args.ignoring:
        if args.net == 'resnet18':
            from models.resnet_ign import resnet18_ign
            criterion = nn.CrossEntropyLoss(reduction='none')
            net = resnet18_ign(criterion, num_classes=num_classes, num_train=num_train)

    else:
        if args.net == 'vgg16':
            from models.vgg import vgg16_bn
            net = vgg16_bn()
        elif args.net == 'vgg13':
            from models.vgg import vgg13_bn
            net = vgg13_bn()
        elif args.net == 'vgg11':
            from models.vgg import vgg11_bn
            net = vgg11_bn()
        elif args.net == 'vgg19':
            from models.vgg import vgg19_bn
            net = vgg19_bn()
        elif args.net == 'densenet121':
            from models.densenet import densenet121
            net = densenet121()
        elif args.net == 'densenet161':
            from models.densenet import densenet161
            net = densenet161()
        elif args.net == 'densenet169':
            from models.densenet import densenet169
            net = densenet169()
        elif args.net == 'densenet201':
            from models.densenet import densenet201
            net = densenet201()
        elif args.net == 'googlenet':
            from models.googlenet import googlenet
            net = googlenet()
        elif args.net == 'inceptionv3':
            from models.inceptionv3 import inceptionv3
            net = inceptionv3()
        elif args.net == 'inceptionv4':
            from models.inceptionv4 import inceptionv4
            net = inceptionv4()
        elif args.net == 'inceptionresnetv2':
            from models.inceptionv4 import inception_resnet_v2
            net = inception_resnet_v2()
        elif args.net == 'xception':
            from models.xception import xception
            net = xception()
        elif args.net == 'resnet18':
            from models.resnet import resnet18
            net = resnet18(num_classes=num_classes)
        elif args.net == 'resnet34':
            from models.resnet import resnet34
            net = resnet34(num_classes=num_classes)
        elif args.net == 'resnet50':
            from models.resnet import resnet50
            net = resnet50(num_classes=num_classes)
        elif args.net == 'resnet101':
            from models.resnet import resnet101
            net = resnet101(num_classes=num_classes)
        elif args.net == 'resnet152':
            from models.resnet import resnet152
            net = resnet152(num_classes=num_classes)
        elif args.net == 'preactresnet18':
            from models.preactresnet import preactresnet18
            net = preactresnet18()
        elif args.net == 'preactresnet34':
            from models.preactresnet import preactresnet34
            net = preactresnet34()
        elif args.net == 'preactresnet50':
            from models.preactresnet import preactresnet50
            net = preactresnet50()
        elif args.net == 'preactresnet101':
            from models.preactresnet import preactresnet101
            net = preactresnet101()
        elif args.net == 'preactresnet152':
            from models.preactresnet import preactresnet152
            net = preactresnet152()
        elif args.net == 'resnext50':
            from models.resnext import resnext50
            net = resnext50()
        elif args.net == 'resnext101':
            from models.resnext import resnext101
            net = resnext101()
        elif args.net == 'resnext152':
            from models.resnext import resnext152
            net = resnext152()
        elif args.net == 'shufflenet':
            from models.shufflenet import shufflenet
            net = shufflenet()
        elif args.net == 'shufflenetv2':
            from models.shufflenetv2 import shufflenetv2
            net = shufflenetv2()
        elif args.net == 'squeezenet':
            from models.squeezenet import squeezenet
            net = squeezenet()
        elif args.net == 'mobilenet':
            from models.mobilenet import mobilenet
            net = mobilenet()
        elif args.net == 'mobilenetv2':
            from models.mobilenetv2 import mobilenetv2
            net = mobilenetv2()
        elif args.net == 'nasnet':
            from models.nasnet import nasnet
            net = nasnet()
        elif args.net == 'attention56':
            from models.attention import attention56
            net = attention56()
        elif args.net == 'attention92':
            from models.attention import attention92
            net = attention92()
        elif args.net == 'seresnet18':
            from models.senet import seresnet18
            net = seresnet18()
        elif args.net == 'seresnet34':
            from models.senet import seresnet34
            net = seresnet34()
        elif args.net == 'seresnet50':
            from models.senet import seresnet50
            net = seresnet50()
        elif args.net == 'seresnet101':
            from models.senet import seresnet101
            net = seresnet101()
        elif args.net == 'seresnet152':
            from models.senet import seresnet152
            net = seresnet152()

        else:
            print('the network name you have entered is not supported yet')
            sys.exit()

    if use_gpu:
        net = net.cuda()

    return net


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader, len(cifar100_training)


def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle 
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data
    
    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = np.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = np.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = np.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def copy_state_dict(net, checkpoint, parallel=False, prefix=''):
    pre_state_dict = checkpoint['state_dict']

    if parallel:
        cur_state_dict = net.module.state_dict()
    else:
        cur_state_dict = net.state_dict()

    for k in cur_state_dict.keys():
        v = _get_params(k, pre_state_dict, prefix=prefix)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue

    net.alphas = checkpoint['alpha_state_dict']
    net.alphas_parameters=[net.alphas]


def copy_optimizer_state_dict(cur_state_dict, pre_state_dict, prefix=''):
    def _get_params(key):
        key = prefix + key
        if key in pre_state_dict:
            return pre_state_dict[key]
        return None

    for k in cur_state_dict.keys():
        v = _get_params(k)
        try:
            if v is None:
                print('parameter {} not found'.format(k))
                continue
            cur_state_dict[k].copy_(v)
        except:
            print('copy param {} failed'.format(k))
            continue


def _get_params(key, pre_state_dict, prefix=''):
    key = prefix + key
    if key in pre_state_dict:
        return pre_state_dict[key]
    return None
