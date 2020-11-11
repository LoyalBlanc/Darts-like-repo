import torch
import torch.nn as nn
import torch.nn.functional as f


@torch.jit.script
def mish(x):
    """
        Applies the mish function element-wise:
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
        See additional documentation for mish class.
    :param x:
    :return:
    """
    return x * torch.tanh(f.softplus(x))


ACTIVATE = nn.ReLU(inplace=False)

OPS = {
    'skip_connect': lambda channel, stride: nn.Identity() if stride == 1 else ReduceConv(channel, channel, stride),
    'dil_conv_3x3': lambda channel, stride: DilationConv(channel, channel, 3, stride, 2),
    'dil_conv_5x5': lambda channel, stride: DilationConv(channel, channel, 5, stride, 2),
    'sep_conv_3x3': lambda channel, stride: SeparableConv(channel, channel, 3, stride),
    'sep_conv_5x5': lambda channel, stride: SeparableConv(channel, channel, 5, stride),
    'max_pool_3x3': lambda channel, stride: MaxPooling(channel, 3, stride),
    'avg_pool_3x3': lambda channel, stride: AvgPooling(channel, 5, stride),
}

PRIMITIVES = [
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'max_pool_3x3',
    'avg_pool_3x3',
]


class ReLUConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation=1, groups=1):
        super(ReLUConvBN, self).__init__()

        self.activate = ACTIVATE
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                              padding=dilation * (kernel_size // 2), dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return self.bn(self.conv(self.activate(x)))


class ReduceConv(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ReduceConv, self).__init__()
        assert out_channel % 2 == 0

        self.activate = ACTIVATE
        self.conv1 = nn.Conv2d(in_channel, out_channel // 2, 1, stride, bias=False)
        self.conv2 = nn.Conv2d(in_channel, out_channel // 2, 1, stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.activate(x)
        x = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        return self.bn(x)


class DilationConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, dilation):
        super(DilationConv, self).__init__()
        self.activate = ACTIVATE
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride,
                               padding=dilation * (kernel_size // 2), dilation=dilation, bias=False)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        return self.bn(self.conv2(self.conv1(self.activate(x))))


class SeparableConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(SeparableConv, self).__init__()
        self.activate = ACTIVATE
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, in_channel, 1, bias=False),
            nn.BatchNorm2d(in_channel)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        return self.conv2(self.activate(self.conv1(self.activate(x))))


class MaxPooling(nn.Module):
    def __init__(self, channel, kernel_size, stride):
        super(MaxPooling, self).__init__()
        self.pooling = nn.MaxPool2d(kernel_size, stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        return self.bn(self.pooling(x))


class AvgPooling(nn.Module):
    def __init__(self, channel, kernel_size, stride):
        super(AvgPooling, self).__init__()
        self.pooling = nn.AvgPool2d(kernel_size, stride, padding=kernel_size // 2, count_include_pad=False)
        self.bn = nn.BatchNorm2d(channel)

    def forward(self, x):
        return self.bn(self.pooling(x))
