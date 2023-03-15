import torch
import torch.nn as nn
from collections import OrderedDict

class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1), _DenseLayer(in_channels+growth_rate*i, growth_rate, bn_size))

class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels, pool=False):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False))
        if pool:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))

class DenseConv_Maker(nn.Module):
    def __init__(self, input_channel, output_channel, growth_rate=12, block_config=(6,12,24,16), bn_size=4, theta=0.5):
        super(DenseConv_Maker, self).__init__()

        num_init_feature = 2 * growth_rate

        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(input_channel, num_init_feature, kernel_size=3, stride=1, padding=1, bias=False)),
        ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i+1), _DenseBlock(num_layers, num_feature, bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i + 1), _Transition(num_feature, int(num_feature * theta), i%2==0))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('conv_final', nn.Conv2d(num_feature, output_channel, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((4, 4)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        return features

