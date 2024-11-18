import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# modules.py

#Nick Allison - Added code
#########################################################################################
class MobileNet2d(nn.Module):
    def __init__(self, out_dim=128, width_mult=1.0):
        super(MobileNet2d, self).__init__()
        self.model = MobileNetV2(out_dim=out_dim, width_mult=width_mult)

    def forward(self, x):
        return self.model(x)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = kernel_size // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # Expand
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        # Depthwise
        layers.extend([
            ConvBNReLU(
                hidden_dim, hidden_dim, stride=stride, groups=hidden_dim
            ),
            # Pointwise
            nn.Conv2d(
                hidden_dim, oup, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(oup),
        ])

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, out_dim=128, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32

        inverted_residual_setting = [
            # t (expansion factor), c (output channels), n (number of blocks), s (stride)
            [1, 16, 1, 1],   # Stage 1
            [6, 24, 2, 2],   # Stage 2
            [6, 32, 3, 2],   # Stage 3
            [6, 64, 4, 2],   # Stage 4
            [6, 96, 3, 1],   # Stage 5
            [6, 160, 3, 2],  # Stage 6
            [6, 320, 1, 1],  # Stage 7
        ]

        # First layer
        input_channel = int(input_channel * width_mult)
        features = [ConvBNReLU(3, input_channel, stride=2)]

        # Inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride_layer = s if i == 0 else 1
                features.append(
                    block(input_channel, output_channel, stride_layer, expand_ratio=t)
                )
                input_channel = output_channel

        # Last convolution
        features.append(ConvBNReLU(input_channel, input_channel, kernel_size=1))

        self.features = nn.Sequential(*features)

        # Adjust output channels to match out_dim
        self.final_conv = nn.Conv2d(input_channel, out_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.features(x)
        x = self.final_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x  # Return feature maps with spatial dimensions
#Nick Allison - End of added code
########################################################################################################

class Cnn3d(torch.nn.Module):
    def __init__(self, in_c):
        super().__init__()

        channels = [64, 64, 128, 64, 64]

        self.stem = torch.nn.Sequential(
            ConvBnRelu3d(in_c, channels[0], ks=1, padding=0),
            ResBlock3d(channels[0]),
        )
        self.conv1x1_1 = ConvBnRelu3d(channels[0], channels[1])
        self.down1 = torch.nn.Sequential(
            ResBlock3d(channels[1]),
            ResBlock3d(channels[1]),
        )
        self.conv1x1_2 = ConvBnRelu3d(channels[1], channels[2])
        self.down2 = torch.nn.Sequential(
            ResBlock3d(channels[2]),
            ResBlock3d(channels[2]),
        )
        self.up1 = torch.nn.Sequential(
            ConvBnRelu3d(channels[2] + channels[1], channels[3]),
            ResBlock3d(channels[3]),
            ResBlock3d(channels[3]),
        )
        self.up2 = torch.nn.Sequential(
            ConvBnRelu3d(channels[3] + channels[0], channels[4]),
            ResBlock3d(channels[4]),
            ResBlock3d(channels[4]),
        )
        self.up3 = torch.nn.Sequential(
            ConvBnRelu3d(channels[4] + in_c, channels[4]),
            ResBlock3d(channels[4]),
            ResBlock3d(channels[4]),
        )
        self.out_c = channels[4]

    def forward(self, x, _):
        x0 = self.stem(x)
        x1 = torch.nn.functional.max_pool3d(self.conv1x1_1(x0), 2)
        x1 = self.down1(x1)
        out = torch.nn.functional.max_pool3d(self.conv1x1_2(x1), 2)
        out = self.down2(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode="nearest")
        out = torch.cat((out, x1), dim=1)
        out = self.up1(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode="nearest")
        out = torch.cat((out, x0), dim=1)
        out = self.up2(out)
        out = torch.cat((out, x), dim=1)
        out = self.up3(out)
        return out


class Cnn2d(torch.nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()

        channel_mean = [0.485, 0.456, 0.406]
        channel_std = [0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(channel_mean, channel_std)

        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        backbone = torchvision.models.efficientnet_v2_s(weights=weights, progress=True)

        self.conv0 = backbone.features[:3]
        self.conv1 = backbone.features[3]
        self.conv2 = backbone.features[4]

        self.out0 = torch.nn.Sequential(
            torch.nn.Conv2d(48, out_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.LeakyReLU(True),
        )

        self.out1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, out_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.LeakyReLU(True),
        )

        self.out2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, out_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.LeakyReLU(True),
        )

        self.out3 = ResBlock2d(out_dim)

    def forward(self, x):
        x = self.normalize(x)

        x = self.conv0(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        x = self.out0(x)
        conv1 = self.out1(conv1)
        conv2 = self.out2(conv2)

        conv1 = torch.nn.functional.interpolate(
            conv1, scale_factor=2, mode="bilinear", align_corners=False
        )
        conv2 = torch.nn.functional.interpolate(
            conv2, scale_factor=4, mode="bilinear", align_corners=False
        )

        x += conv1
        x += conv2

        return self.out3(x)


class FeatureFusion(torch.nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.out_c = in_c
        self.bn = torch.nn.BatchNorm3d(self.out_c)

    def forward(self, x, valid):
        counts = torch.sum(valid, dim=1, keepdim=True)
        counts.masked_fill_(counts == 0, 1)
        x.masked_fill_(~valid[:, :, None], 0)
        x /= counts[:, :, None]
        mean = x.sum(dim=1)

        return self.bn(mean)


class ResBlock(torch.nn.Module):
    def forward(self, x):
        out = self.net(x)
        out += x
        torch.nn.functional.leaky_relu_(out)
        return out


class ResBlock3d(ResBlock):
    def __init__(self, c, ks=3, padding=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(c, c, ks, bias=False, padding=padding),
            torch.nn.BatchNorm3d(c),
            torch.nn.LeakyReLU(True),
            torch.nn.Conv3d(c, c, ks, bias=False, padding=padding),
            torch.nn.BatchNorm3d(c),
        )


class ResBlock2d(ResBlock):
    def __init__(self, c, ksize=3, padding=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(c, c, ksize, bias=False, padding=padding),
            torch.nn.BatchNorm2d(c),
            torch.nn.LeakyReLU(True),
            torch.nn.Conv2d(c, c, ksize, bias=False, padding=padding),
            torch.nn.BatchNorm2d(c),
        )


class ResBlock1d(ResBlock):
    def __init__(self, c):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(c, c, 1, bias=False),
            torch.nn.BatchNorm1d(c),
            torch.nn.LeakyReLU(True),
            torch.nn.Conv1d(c, c, 1, bias=False),
            torch.nn.BatchNorm1d(c),
        )


class ConvBnRelu3d(torch.nn.Module):
    def __init__(self, in_c, out_c, ks=3, padding=1):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(in_c, out_c, ks, padding=padding, bias=False),
            torch.nn.BatchNorm3d(out_c),
            torch.nn.LeakyReLU(True),
        )

    def forward(self, x):
        return self.net(x)
