import torch
import torch.nn as nn
from collections import OrderedDict

# nn.ConvTranspose2d(in_channels=output_channel[2], out_channels=output_channel[2], kernel_size=4, stride=2, padding=1),
class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('deconv1', nn.ConvTranspose2d(in_channels=in_channels, out_channels=bn_size*growth_rate, kernel_size=3, stride=1, padding=1))
        self.add_module('norm2', nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('deconv2', nn.ConvTranspose2d(in_channels=bn_size*growth_rate, out_channels=growth_rate, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)

class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i+1), _DenseLayer(in_channels+growth_rate*i, growth_rate, bn_size))

class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('deconv_up', nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1))


class DenseDeconv_Maker(nn.Module):
    def __init__(self, input_channel, output_channel, growth_rate, block_config, bn_size, theta):
        super(DenseDeconv_Maker, self).__init__()

        num_feature = input_channel
        self.features = nn.Sequential(OrderedDict([]))

        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i + 1), _DenseBlock(num_layers, num_feature, bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.add_module('transition%d' % (i + 1), _Transition(num_feature, int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('conv_final', nn.Conv2d(num_feature, output_channel, kernel_size=3, stride=1, padding=1, bias=False))
        self.features.add_module('relu5', nn.ReLU(True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((32,32)))
        self.features.add_module('tanh', nn.Tanh())

    def forward(self, x):
        features = self.features(x)
        out = features
        return out



