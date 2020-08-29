"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

epsilon = 1e-10


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, criterion, num_classes=100, num_train=0, softmax=True, isalpha=True):
        super().__init__()

        self.block = block
        self.num_block = num_block
        self.num_train = num_train
        self._criterion = criterion
        self.num_classes = num_classes

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.softmax = softmax
        if not self.softmax:
            print('no softmax for alphas')
        self.isalpha = isalpha
        if not self.isalpha:
            print('no alpha acummulated')

        self._initialize_alphas()

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

    def new(self):
        model_new = ResNet(self.block, self.num_block, self._criterion, self.num_classes, self.num_train)
        for x, y in zip(model_new.alphas_parameters, self._alphas_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _loss(self, input, label, indices=None):
        pred = self(input)
        losses = self._criterion(pred, label)
        # print(torch.max(losses), torch.min(losses))
        if indices is not None:
            alphas = self.alphas
            if self.softmax:
                alphas = F.softmax(alphas)
                alphas = alphas[indices] * self.num_train / len(indices)
            else:
                # alphas = torch.sigmoid(alphas[indices])
                alphas=alphas[indices]
                # alphas = alphas[indices] / (torch.sum(torch.abs(alphas[indices])) + epsilon)
            loss = losses.dot(alphas)
        else:
            loss = torch.mean(losses)
        return loss

    def criterion(self, outputs, label, indices=None):
        losses = self._criterion(outputs, label)
        if indices is not None:
            alphas = self.alphas
            if self.softmax:
                alphas = F.softmax(alphas)
                alphas = alphas[indices] * self.num_train / len(indices)
            else:
                alphas = alphas[indices] / (torch.sum(torch.abs(alphas[indices])) + epsilon)
            loss = losses.dot(alphas)
        else:
            loss = torch.mean(losses)
        return loss

    def _initialize_alphas(self):
        if self.softmax:
            self.alphas = Variable(1e-3 * torch.zeros(self.num_train).cuda(), requires_grad=True)
        else:
            self.alphas = Variable(torch.ones(self.num_train).cuda(), requires_grad=True)
        self.alphas_parameters = [self.alphas]

    def _alphas_parameters(self):
        return [self.alphas]

    def _update_alphas(self, alphas):
        self.alphas = alphas
        self.alphas.requires_grad_()


def resnet18_ign(criterion, num_classes=100, num_train=0, softmax=True, isalpha=True):
    """ return a ResNet 18 object
    """

    return ResNet(BasicBlock, [2, 2, 2, 2], criterion, num_classes=num_classes, num_train=num_train, softmax=softmax,
                  isalpha=isalpha)


def resnet34(num_classes=100):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=100):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=100):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes)


def resnet152(num_classes=100):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3], num_classes=num_classes)
