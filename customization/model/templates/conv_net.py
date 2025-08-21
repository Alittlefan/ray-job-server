from typing import Optional, Sequence, Type

import torch
from torch import nn


class SquashDims(nn.Module):
    def __init__(self, ndims_in: int = 3):
        super().__init__()
        self.ndims_in = ndims_in

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.flatten(-self.ndims_in, -1)
        return value


class ConvNet(nn.Sequential):
    def __init__(
        self,
        in_channels: Optional[int],
        num_channels: Sequence[int] = (),
        kernel_sizes: Sequence[int] = (),
        strides: Sequence[int] = (),
        paddings: Sequence[int] = (),
        activation_class: Optional[Type[nn.Module]] = nn.Tanh,
        activation_kwargs: Optional[dict] = None,
        norm_class: Optional[Type[nn.Module]] = None,
        norm_kwargs: Optional[dict] = None,
        squash_last_layer: bool = True,
    ):
        assert len(num_channels) == len(
            kernel_sizes
        ), f"kernel_sizes, strides, paddings must have same length, but got {len(num_channels)} and {len(kernel_sizes)}."
        assert len(kernel_sizes) == len(
            strides
        ), f"kernel_sizes, strides, paddings must have same length, but got {len(kernel_sizes)} and {len(strides)}."
        assert len(strides) == len(
            paddings
        ), f"kernel_sizes, strides, paddings must have same length, but got {len(strides)} and {len(paddings)}."
        if not activation_kwargs:
            activation_kwargs = {}
        layers = []
        _in = in_channels
        for _out, _kernel, _stride, _padding in zip(
            num_channels, kernel_sizes, strides, paddings
        ):
            if _in:
                layers.append(
                    nn.Conv2d(
                        _in,
                        _out,
                        kernel_size=_kernel,
                        stride=_stride,
                        padding=_padding,
                    )
                )
            else:
                layers.append(
                    nn.LazyConv2d(
                        _out,
                        kernel_size=_kernel,
                        stride=_stride,
                        padding=_padding,
                    )
                )
            if activation_class:
                layers.append(activation_class(**activation_kwargs))
            if norm_class:
                layers.append(norm_class(**norm_kwargs))
            _in = _out
        if squash_last_layer:
            layers.append(SquashDims())
        super().__init__(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, C, L, W = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        out = super(ConvNet, self).forward(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out
