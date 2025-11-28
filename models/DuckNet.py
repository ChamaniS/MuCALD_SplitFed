import torch
import torch.nn as nn

from .layers import ConvBlock, ResidualBlock, DUCKBlock


class DuckNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, num_filters: int = 34):
        super(DuckNet, self).__init__()

        self.conv11 = ConvBlock(input_channels, num_filters * 2, 2, 2, 0, is_relu=False, is_bn=False)
        self.conv21 = ConvBlock(num_filters * 2, num_filters * 4, 2, 2, 0, is_relu=False, is_bn=False)
        self.conv31 = ConvBlock(num_filters * 4, num_filters * 8, 2, 2, 0, is_relu=False, is_bn=False)
        self.conv41 = ConvBlock(num_filters * 8, num_filters * 16, 2, 2, 0, is_relu=False, is_bn=False)
        self.conv51 = ConvBlock(num_filters * 16, num_filters * 32, 2, 2, 0, is_relu=False, is_bn=False)

        self.duck01 = DUCKBlock(input_channels, num_filters)
        self.duck11 = DUCKBlock(num_filters * 2, num_filters * 2)
        self.duck21 = DUCKBlock(num_filters * 4, num_filters * 4)
        self.duck31 = DUCKBlock(num_filters * 8, num_filters * 8)
        self.duck41 = DUCKBlock(num_filters * 16, num_filters * 16)

        self.conv12 = ConvBlock(num_filters, num_filters * 2, 2, 2, 0, is_relu=False, is_bn=False)
        self.conv22 = ConvBlock(num_filters * 2, num_filters * 4, 2, 2, 0, is_relu=False, is_bn=False)
        self.conv32 = ConvBlock(num_filters * 4, num_filters * 8, 2, 2, 0, is_relu=False, is_bn=False)
        self.conv42 = ConvBlock(num_filters * 8, num_filters * 16, 2, 2, 0, is_relu=False, is_bn=False)
        self.conv52 = ConvBlock(num_filters * 16, num_filters * 32, 2, 2, 0, is_relu=False, is_bn=False)

        self.r51 = ResidualBlock(num_filters * 32, num_filters * 32)
        self.r52 = ResidualBlock(num_filters * 32, num_filters * 16)

        self.upconv5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upconv4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upconv3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upconv2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upconv1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.duck02 = DUCKBlock(num_filters, num_filters)
        self.duck12 = DUCKBlock(num_filters * 2, num_filters)
        self.duck22 = DUCKBlock(num_filters * 4, num_filters * 2)
        self.duck32 = DUCKBlock(num_filters * 8, num_filters * 4)
        self.duck42 = DUCKBlock(num_filters * 16, num_filters * 8)

        self.last_conv = nn.Sequential(
            nn.Conv2d(num_filters, num_classes, 1, 1, 0),
            #nn.Sigmoid()
        )

    def forward(self, x):
        p1 = self.conv11(x)
        p2 = self.conv21(p1)
        p3 = self.conv31(p2)
        p4 = self.conv41(p3)
        p5 = self.conv51(p4)

        t0 = self.duck01(x)

        li1 = self.conv12(t0)
        s1 = p1 + li1
        t1 = self.duck11(s1)

        li2 = self.conv22(t1)
        s2 = p2 + li2
        t2 = self.duck21(s2)

        li3 = self.conv32(t2)
        s3 = p3 + li3
        t3 = self.duck31(s3)

        li4 = self.conv42(t3)
        s4 = p4 + li4
        t4 = self.duck41(s4)

        li5 = self.conv52(t4)
        s5 = p5 + li5
        t51 = self.r51(s5)
        t52 = self.r52(t51)

        u4 = self.upconv5(t52)
        c4 = u4 + t4
        q4 = self.duck42(c4)

        u3 = self.upconv4(q4)
        c3 = u3 + t3
        q3 = self.duck32(c3)

        u2 = self.upconv3(q3)
        c2 = u2 + t2
        q2 = self.duck22(c2)

        u1 = self.upconv2(q2)
        c1 = u1 + t1
        q1 = self.duck12(c1)

        u0 = self.upconv1(q1)
        c0 = u0 + t0
        q0 = self.duck02(c0)

        out = self.last_conv(q0)

        return out


if __name__ == '__main__':
    model = DuckNet(3, 1)
    dummy = torch.randn(4, 3, 352, 352)
    out = model(dummy)
    print(out.shape)