import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding, dilation=1, is_relu=True,
                 is_bn=True):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding, dilation)
        self.relu = nn.ReLU() if is_relu else None
        self.bn = nn.BatchNorm2d(output_channel) if is_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = self.relu(x)
        if self.bn:
            x = self.bn(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(ResidualBlock, self).__init__()

        self.conv1 = ConvBlock(input_channel, output_channel, 1, 1, 0, is_relu=True, is_bn=False)
        self.conv2 = ConvBlock(input_channel, output_channel, 3, 1, 1, is_relu=True, is_bn=True)
        self.conv3 = ConvBlock(output_channel, output_channel, 3, 1, 1, is_relu=True, is_bn=True)
        self.bn = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        residual = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual
        x = self.bn(x)
        return x


class MidScopeBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(MidScopeBlock, self).__init__()

        self.conv1 = ConvBlock(input_channel, output_channel, 3, 1, 1, is_relu=True, is_bn=True)
        self.conv2 = ConvBlock(output_channel, output_channel, 3, 1, 2, dilation=2, is_relu=True, is_bn=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class WideScopeBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(WideScopeBlock, self).__init__()

        self.conv1 = ConvBlock(input_channel, output_channel, 3, 1, 1, is_relu=True, is_bn=True)
        self.conv2 = ConvBlock(output_channel, output_channel, 3, 1, 2, dilation=2, is_relu=True, is_bn=True)
        self.conv3 = ConvBlock(output_channel, output_channel, 3, 1, 3, dilation=3, is_relu=True, is_bn=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class SeparatedBlock(nn.Module):
    def __init__(self, input_channel, output_channel, size=3):
        super(SeparatedBlock, self).__init__()

        if size % 2 == 0:
            self.padding1 = (0, 0, size // 2 - 1, size // 2)
            self.padding2 = (size // 2 - 1, size // 2, 0, 0)
        else:
            self.padding1 = (0, size // 2, 0, size // 2)
            self.padding2 = (size // 2, 0, size // 2, 0)

        self.conv1 = ConvBlock(input_channel, output_channel, (1, size), 1, 0, is_relu=True, is_bn=True)
        self.conv2 = ConvBlock(output_channel, output_channel, (size, 1), 1, 0, is_relu=True, is_bn=True)

    def forward(self, x):
        x = F.pad(x, self.padding1)
        x = self.conv1(x)
        x = F.pad(x, self.padding2)
        x = self.conv2(x)
        return x


class DUCKBlock(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DUCKBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(input_channel)

        self.wide_scope = WideScopeBlock(input_channel, output_channel)
        self.mid_scope = MidScopeBlock(input_channel, output_channel)
        self.separated = SeparatedBlock(input_channel, output_channel, size=6)
        self.residual_1 = ResidualBlock(input_channel, output_channel)
        self.residual_2 = nn.Sequential(
            ResidualBlock(input_channel, output_channel),
            ResidualBlock(output_channel, output_channel)
        )
        self.residual_3 = nn.Sequential(
            ResidualBlock(input_channel, output_channel),
            ResidualBlock(output_channel, output_channel),
            ResidualBlock(output_channel, output_channel)
        )
        self.bn2 = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        x = self.bn1(x)
        x1 = self.wide_scope(x)
        x2 = self.mid_scope(x)
        x3 = self.separated(x)
        x4 = self.residual_1(x)
        x5 = self.residual_2(x)
        x6 = self.residual_3(x)
        x = x1 + x2 + x3 + x4 + x5 + x6
        x = self.bn2(x)
        return x