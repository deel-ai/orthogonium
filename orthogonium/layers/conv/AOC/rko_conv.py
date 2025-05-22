import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_2_t

from orthogonium.layers.conv.AOC.fast_block_ortho_conv import conv_singular_values_numpy
from orthogonium.layers.conv.AOC.padding_convtranspose import PaddingConvTranspose2d
from orthogonium.reparametrizers import OrthoParams


class RKOParametrizer(nn.Module):
    def __init__(
        self, kernel_shape, groups, scale, ortho_params: OrthoParams = OrthoParams()
    ):
        super(RKOParametrizer, self).__init__()
        self.kernel_shape = kernel_shape
        self.groups = groups
        out_channels, in_channels, k1, k2 = kernel_shape
        in_channels *= groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k1 = k1
        self.k2 = k2
        self.scale = scale
        self.register_module(
            "pi",
            ortho_params.spectral_normalizer(
                weight_shape=(
                    self.groups,
                    out_channels // self.groups,
                    in_channels // groups * k1 * k2,
                ),
            ),
        )
        self.register_module(
            "bjorck",
            ortho_params.orthogonalizer(
                weight_shape=(
                    self.groups,
                    out_channels // self.groups,
                    in_channels // groups * k1 * k2,
                ),
            ),
        )

    def forward(self, X):
        X = X.view(
            self.groups,
            self.out_channels // self.groups,
            (self.in_channels // self.groups) * self.k1 * self.k2,
        )
        X = self.pi(X)
        X = self.bjorck(X)
        X = X.reshape(
            self.out_channels, self.in_channels // self.groups, self.k1, self.k2
        )
        return X * self.scale

    def right_inverse(self, X):
        return X


def attach_rko_weight(
    layer,
    weight_name,
    kernel_shape,
    groups,
    scale=None,
    ortho_params: OrthoParams = OrthoParams(),
):
    out_channels, in_channels, kernel_size, k2 = kernel_shape
    in_channels *= groups
    if scale is None:
        scale = 1 / math.sqrt(kernel_size * kernel_size)

    layer.register_parameter(
        weight_name,
        torch.nn.Parameter(
            torch.Tensor(torch.randn(*kernel_shape)),
            requires_grad=True,
        ),
    )
    weight = getattr(layer, weight_name)
    torch.nn.init.normal_(weight)
    parametrize.register_parametrization(
        layer,
        weight_name,
        RKOParametrizer(
            kernel_shape,
            groups,
            scale,
            ortho_params=ortho_params,
        ),
    )


class RKOConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = "same",
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "circular",
        ortho_params: OrthoParams = OrthoParams(),
    ):
        super(RKOConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        if (
            True  # (self.out_channels >= self.in_channels) # investigate why it don't work
            and (((self.dilation[0] % self.stride[0]) == 0) and (self.stride[0] > 1))
            and (((self.dilation[1] % self.stride[1]) == 0) and (self.stride[1] > 1))
        ):
            raise ValueError(
                "dilation must be 1 when stride is not 1. The set of orthogonal convolutions is empty in this setting."
            )
        torch.nn.init.orthogonal_(self.weight)
        self.scale = 1 / math.sqrt(
            math.ceil(self.kernel_size[0] / self.stride[0])
            * math.ceil(self.kernel_size[1] / self.stride[1])
        )
        parametrize.register_parametrization(
            self,
            "weight",
            RKOParametrizer(
                kernel_shape=(
                    out_channels,
                    in_channels // self.groups,
                    self.kernel_size[0],
                    self.kernel_size[1],
                ),
                groups=self.groups,
                scale=self.scale,
                ortho_params=ortho_params,
            ),
        )


class RkoConvTranspose2d(PaddingConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_2_t = 1,
        padding_mode: str = "zeros",
        ortho_params: OrthoParams = OrthoParams(),
    ):
        super(RkoConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        if self.kernel_size[0] < self.stride[0] or self.kernel_size[1] < self.stride[1]:
            raise ValueError(
                "kernel size must be smaller than stride. The set of orthogonal convolutions is empty in this setting."
            )
        if (
            (self.out_channels <= self.in_channels)
            and (((self.dilation[0] % self.stride[0]) == 0) and (self.stride[0] > 1))
            and (((self.dilation[1] % self.stride[1]) == 0) and (self.stride[1] > 1))
        ):
            raise ValueError(
                "dilation must be 1 when stride is not 1. The set of orthonal convolutions is empty in this setting."
            )
        if (
            self.stride[0] != self.kernel_size[0]
            or self.stride[1] != self.kernel_size[1]
        ):
            self.scale = 1 / math.sqrt(
                math.ceil(self.kernel_size[0] / self.stride[0])
                * math.ceil(self.kernel_size[1] / self.stride[1])
            )
        else:
            self.scale = 1
        del self.weight
        attach_rko_weight(
            self,
            "weight",
            (in_channels, out_channels // groups, self.stride[0], self.stride[1]),
            groups,
            scale=self.scale,
            ortho_params=ortho_params,
        )
