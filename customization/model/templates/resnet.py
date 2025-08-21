from typing import Optional, Type

import torch
import torch.nn as nn


class ResNetBlock(nn.Module):
    def __init__(
        self,
        num_ch: int,
        activation_class: Optional[Type[nn.Module]] = nn.Tanh,
        activation_kwargs: Optional[dict] = None,
    ):
        super(ResNetBlock, self).__init__()
        if not activation_kwargs:
            activation_kwargs = {}
        resnet_block = []
        if activation_class:
            resnet_block.append(activation_class(**activation_kwargs))
        resnet_block.append(
            nn.LazyConv2d(
                out_channels=num_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        if activation_class:
            resnet_block.append(activation_class(**activation_kwargs))
        resnet_block.append(
            nn.Conv2d(
                in_channels=num_ch,
                out_channels=num_ch,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.seq = nn.Sequential(*resnet_block)

    def forward(self, x):
        y = x + self.seq(x)
        return y


class ConvNetBlock(nn.Module):
    def __init__(
        self,
        num_ch: int,
        kernel_size=3,
        stride=1,
        max_stride=2,
        padding=1,
        activation_class: Optional[Type[nn.Module]] = nn.Tanh,
        activation_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        if not activation_kwargs:
            activation_kwargs = {}
        conv = nn.LazyConv2d(
            out_channels=num_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        mp = nn.MaxPool2d(kernel_size=kernel_size, stride=max_stride, padding=padding)
        self.feats_conv = nn.Sequential(conv, mp)
        self.resnet1 = ResNetBlock(
            num_ch=num_ch,
            activation_class=activation_class,
            activation_kwargs=activation_kwargs,
        )
        self.resnet2 = ResNetBlock(
            num_ch=num_ch,
            activation_class=activation_class,
            activation_kwargs=activation_kwargs,
        )

    def forward(self, x):
        x = self.feats_conv(x)
        x = self.resnet1(x)
        x = self.resnet1(x)
        return x


class ImpalaNet(nn.Sequential):
    def __init__(
        self,
        channels=(16, 32, 32),
        activation_class: Optional[Type[nn.Module]] = nn.Tanh,
        activation_kwargs: Optional[dict] = None,
    ):
        layers = [
            ConvNetBlock(
                num_ch,
                activation_class=activation_class,
                activation_kwargs=activation_kwargs,
            )
            for num_ch in channels
        ]
        if activation_class:
            layers += [activation_class(**activation_kwargs)]
        super().__init__(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, C, L, W = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        out = super(ImpalaNet, self).forward(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out
