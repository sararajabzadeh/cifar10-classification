import torch.nn as nn


class Block(nn.Module):

    def __init__(self, in_channels, out_channels, downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        out = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            out = self.downsample(out)
        x += out
        x = self.relu(x)
        return x


class Resnet18(nn.Module):

    def __init__(self, image_channels, num_classes):
        super(Resnet18, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, padding=3, stride=2)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.first = self.make_layers(64, 64, 1)
        self.second = self.make_layers(64, 128, 2)
        self.third = self.make_layers(128, 256, 2)
        self.fourth = self.make_layers512(256, 512, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.first(x)
        x = self.second(x)
        x = self.third(x)
        x = self.fourth(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def make_layers(self, in_channels, out_channels, stride):
        downsample = None

        if stride != 1:
            downsample = self.downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(in_channels, out_channels, downsample=downsample, stride=stride),
            Block(out_channels, out_channels))

    def make_layers512(self, in_channels, out_channels, stride):
        downsample = None

        if stride != 1:
            downsample = self.downsample(in_channels, out_channels)

        return Block(in_channels, out_channels, downsample=downsample, stride=stride)

    def downsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels))
