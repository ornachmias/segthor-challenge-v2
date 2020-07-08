import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils import model_zoo as model_zoo

from models.vision_resnet import model_urls, conv1x1, conv3x3


def resnet101_wo_norm(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetNoNorm(BottleneckNoNorm, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet101'])
        model_state_dict = model.state_dict()
        for key in list(state_dict.keys()):
            if 'fc' in key or 'bn' in key:
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)

    return model


class BottleneckNoNorm(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckNoNorm, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetNoNorm(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNetNoNorm, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #         self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_down0 = self.layer1(x)
        x_down1 = self.layer2(x_down0)
        x_down2 = self.layer3(x_down1)
        x_down3 = self.layer4(x_down2)

        return x_down0, x_down1, x_down2, x_down3


class _DenseLayerNoNorm(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayerNoNorm, self).__init__()
        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module(
            'conv1',
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module(
            'conv2',
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayerNoNorm, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlockNoNorm(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlockNoNorm, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayerNoNorm(num_input_features + i * growth_rate,
                                      growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class ResUNet101NoBatchNorm(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 drop_rate,
                 pretrained,
                 fine_tune=False):
        super(ResUNet101NoBatchNorm, self).__init__()
        from .vision_resnet import resnet101
        self.forward_resnet = resnet101_wo_norm(pretrained=pretrained)
        self.up0 = up_no_norm(
            2048 + 1024,
            1024,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up1 = up_no_norm(
            1024 + 512 + 48,
            512,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up2 = up_no_norm(
            512 + 256 + 48,
            256,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.back_conv = nn.Sequential(_BackwardTransitionNoNorm(256 + 48, 256),
                                       _BackwardTransitionNoNorm(
                                           256, 256))
        self.last_conv = nn.Sequential(
            nn.ReLU(inplace=False), nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.class_conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid())

    def forward(self, x):
        down0, down1, down2, out = self.forward_resnet(x)
        x = self.up0(out, down2)
        x = self.up1(x, down1)
        x = self.up2(x, down0)
        x = self.back_conv(x)
        x_c = self.class_conv(x)
        x_s = self.last_conv(x)
        return x_s, x_c.view(x_c.size(0), x_c.size(1))


class _BackwardTransitionNoNorm(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_BackwardTransitionNoNorm, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(
            x,
            size=(x.shape[2] * 2, x.shape[3] * 2),
            mode='bilinear',
            align_corners=True)
        return x


class up_no_norm(nn.Module):
    def __init__(self,
                 num_features,
                 num_out,
                 num_layers,
                 drop_rate,
                 bn_size=4,
                 growth_rate=32):
        super(up_no_norm, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(num_features, num_out, kernel_size=1, stride=1))

        self.block = _DenseBlockNoNorm(
            num_layers=num_layers,
            num_input_features=num_out,
            bn_size=bn_size,
            growth_rate=growth_rate,
            drop_rate=drop_rate)

    def forward(self, x1, x2):
        x1 = F.interpolate(
            x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.block(x)
        return x
