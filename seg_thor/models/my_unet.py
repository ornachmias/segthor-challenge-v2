import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module(
            'conv1',
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
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
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _BackwardTransition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(_BackwardTransition, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        x = self.conv(x)
        x = F.interpolate(
            x,
            size=(x.shape[2] * 2, x.shape[3] * 2),
            mode='bilinear',
            align_corners=True)
        return x


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class up(nn.Module):
    def __init__(self,
                 num_features,
                 num_out,
                 num_layers,
                 drop_rate,
                 bn_size=4,
                 growth_rate=32):
        super(up, self).__init__()

        self.conv = nn.Sequential(
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_out, kernel_size=1, stride=1))

        self.block = _DenseBlock(
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


class DenseUNet161(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 drop_rate,
                 pretrained,
                 fine_tune=False):
        super(DenseUNet161, self).__init__()
        from .vision_densenet import densenet161
        self.forward_densenet = densenet161(
            pretrained=pretrained, num_classes=n_classes, drop_rate=drop_rate)
        if fine_tune:
            for p in self.parameters():
                p.requires_grad = False
        self.up0 = up(
            2208 + 2112,
            1024,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up1 = up(
            768 + 1024 + 48,
            768,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up2 = up(
            384 + 768 + 48,
            384,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.back_conv = nn.Sequential(_BackwardTransition(384 + 48, 256),
                                       _BackwardTransition(
                                           256, 256))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.class_conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid())

    def forward(self, x):
        down0, down1, down2, out = self.forward_densenet(x)
        x = self.up0(out, down2)
        x = self.up1(x, down1)
        x = self.up2(x, down0)
        x = self.back_conv(x)
        x_c = self.class_conv(x)
        x_s = self.last_conv(x)
        return x_s, x_c.view(x_c.size(0), x_c.size(1))


class DenseUNet201(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 drop_rate,
                 pretrained,
                 fine_tune=False):
        super(DenseUNet201, self).__init__()
        from .vision_densenet import densenet201
        self.forward_densenet = densenet201(
            pretrained=pretrained, num_classes=n_classes, drop_rate=drop_rate)
        if fine_tune:
            for p in self.parameters():
                p.requires_grad = False
        self.up0 = up(
            1792 + 1920,
            1024,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up1 = up(
            512 + 1024 + 48,
            512,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up2 = up(
            256 + 512 + 48,
            256,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.back_conv = nn.Sequential(_BackwardTransition(256 + 48, 256),
                                       _BackwardTransition(
                                           256, 256))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.class_conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid())

    def forward(self, x):
        down0, down1, down2, out = self.forward_densenet(x)
        x = self.up0(out, down2)
        x = self.up1(x, down1)
        x = self.up2(x, down0)
        x = self.back_conv(x)
        x_c = self.class_conv(x)
        x_s = self.last_conv(x)
        return x_s, x_c.view(x_c.size(0), x_c.size(1))


class DenseUNet169(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 drop_rate,
                 pretrained,
                 fine_tune=False):
        super(DenseUNet169, self).__init__()
        from .vision_densenet import densenet169
        self.forward_densenet = densenet169(
            pretrained=pretrained, num_classes=n_classes, drop_rate=drop_rate)
        if fine_tune:
            for p in self.parameters():
                p.requires_grad = False
        self.up0 = up(
            1280 + 1664,
            1024,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up1 = up(
            512 + 1024 + 48,
            512,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up2 = up(
            256 + 512 + 48,
            256,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.back_conv = nn.Sequential(_BackwardTransition(256 + 48, 192),
                                       _BackwardTransition(
                                           192, 192))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True), nn.Conv2d(192, n_classes, kernel_size=1, stride=1))
        self.class_conv = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid())

    def forward(self, x):
        down0, down1, down2, out = self.forward_densenet(x)
        x = self.up0(out, down2)
        x = self.up1(x, down1)
        x = self.up2(x, down0)
        x = self.back_conv(x)
        x_c = self.class_conv(x)
        x_s = self.last_conv(x)
        return x_s, x_c.view(x_c.size(0), x_c.size(1))


class DenseUNet121(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 drop_rate,
                 pretrained,
                 fine_tune=False):
        super(DenseUNet121, self).__init__()
        from .vision_densenet import densenet121
        self.forward_densenet = densenet121(
            pretrained=pretrained, num_classes=n_classes, drop_rate=drop_rate)
        if fine_tune:
            for p in self.parameters():
                p.requires_grad = False
        self.up0 = up(
            1024 + 1024,
            1024,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up1 = up(
            512 + 1024 + 48,
            512,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up2 = up(
            256 + 512 + 48,
            256,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.back_conv = nn.Sequential(_BackwardTransition(256 + 48, 256),
                                       _BackwardTransition(
                                           256, 256))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.class_conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid())

    def forward(self, x):
        down0, down1, down2, out = self.forward_densenet(x)
        x = self.up0(out, down2)
        x = self.up1(x, down1)
        x = self.up2(x, down0)
        x = self.back_conv(x)
        x_c = self.class_conv(x)
        x_s = self.last_conv(x)
        return x_s, x_c.view(x_c.size(0), x_c.size(1))


class ResUNet101(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 drop_rate,
                 pretrained,
                 fine_tune=False):
        super(ResUNet101, self).__init__()
        from .vision_resnet import resnet101
        self.forward_resnet = resnet101(pretrained=pretrained)
        self.up0 = up(
            2048 + 1024,
            1024,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up1 = up(
            1024 + 512 + 48,
            512,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up2 = up(
            512 + 256 + 48,
            256,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.back_conv = nn.Sequential(_BackwardTransition(256 + 48, 256),
                                       _BackwardTransition(
                                           256, 256))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.class_conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
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


class ResUNet101Attention(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 drop_rate,
                 pretrained,
                 fine_tune=False):
        super(ResUNet101Attention, self).__init__()
        from .vision_resnet import resnet101
        self.forward_resnet = resnet101(pretrained=pretrained)

        n1 = 128
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.back_conv = nn.Sequential(_BackwardTransition(filters[1], filters[1]),
                                       _BackwardTransition(filters[1], filters[1]))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True), nn.Conv2d(filters[1], n_classes, kernel_size=1, stride=1))
        self.class_conv = nn.Sequential(
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters[1], 4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid())

    def forward(self, x):
        down0, down1, down2, down3 = self.forward_resnet(x)
        d5 = self.Up5(down3)

        x4 = self.Att5(g=d5, x=down2)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=down1)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=down0)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        out = self.back_conv(d3)
        out_1 = self.last_conv(out)
        out_2 = self.class_conv(out)
        return out_1, out_2.view(out_2.size(0), out_2.size(1))


class ResUNet101Index(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 drop_rate,
                 pretrained,
                 fine_tune=False):
        super(ResUNet101Index, self).__init__()
        from .vision_resnet import resnet101
        self.embed_index = nn.Conv2d(n_channels, 3, kernel_size=1, stride=1)
        self.forward_resnet = resnet101(pretrained=pretrained)
        self.up0 = up(
            2048 + 1024,
            1024,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up1 = up(
            1024 + 512 + 48,
            512,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up2 = up(
            512 + 256 + 48,
            256,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.back_conv = nn.Sequential(_BackwardTransition(256 + 48, 256),
                                       _BackwardTransition(
                                           256, 256))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True), nn.Conv2d(256, n_classes, kernel_size=1, stride=1))
        self.class_conv = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 4, kernel_size=1, stride=1),
            nn.AdaptiveAvgPool2d(1), nn.Sigmoid())

    def forward(self, x):
        x = self.embed_index(x)
        down0, down1, down2, out = self.forward_resnet(x)
        x = self.up0(out, down2)
        x = self.up1(x, down1)
        x = self.up2(x, down0)
        x = self.back_conv(x)
        x_c = self.class_conv(x)
        x_s = self.last_conv(x)
        return x_s, x_c.view(x_c.size(0), x_c.size(1))


class ResUNet152(nn.Module):
    def __init__(self,
                 n_channels,
                 n_classes,
                 drop_rate,
                 pretrained,
                 fine_tune=False):
        super(ResUNet152, self).__init__()
        from .vision_resnet import resnet152
        self.forward_resnet = resnet152(pretrained=pretrained)
        self.up0 = up(
            2048 + 1024,
            1024,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up1 = up(
            1024 + 512 + 48,
            512,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.up2 = up(
            512 + 256 + 48,
            256,
            num_layers=1,
            drop_rate=drop_rate,
            bn_size=4,
            growth_rate=48)
        self.back_conv = nn.Sequential(_BackwardTransition(256 + 48, 256),
                                       _BackwardTransition(
                                           256, 192))
        self.last_conv = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True), nn.Conv2d(192, n_classes, kernel_size=1, stride=1))
        self.class_conv = nn.Sequential(
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 4, kernel_size=1, stride=1),
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
