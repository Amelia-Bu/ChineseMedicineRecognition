import torch
import torch.nn as nn
import utils
from trainPara import train_parameters


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(6400, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.Linear(4096, utils.train_parameters['class_dim'])
        )

    def forward(self, inputs, label=None):
        out04 = self.net(inputs)
        if label is not None:
            # acc = nn.metric.accuracy(input=out, label=label)
            _, predicted = torch.max(input=out04.data, dim=1)
            label = label.squeeze(dim=1)
            acc = (predicted == label).sum().item() / len(label)
            # print('acc', acc)
            return out04, acc
        else:
            return out04


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  # (卷积层数，输出通道数)
        self.in_channels = 3
        self.out_channels = 512
        self.conv_blks = []

        for (num_convs, out_channels) in self.conv_arch:
            self.conv_blks.append(vgg_block(
                num_convs, self.in_channels, out_channels))
            self.in_channels = out_channels

        self.net = nn.Sequential(
            *self.conv_blks,
            nn.Flatten(),
            nn.Linear(self.out_channels * 7 * 7, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(4096, 10)
        )

    def forward(self, inputs, label=None):

        out04 = self.net(inputs)

        if label is not None:
            _, predicted = torch.max(input=out04.data, dim=1)
            label = label.squeeze(dim=1)
            acc = (predicted == label).sum().item() / len(label)
            return out04, acc
        else:
            return out04

