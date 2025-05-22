import warnings
from typing import Union
import math

import numpy as np
from torch import nn as nn
from torch.nn.common_types import _size_2_t
from torch.nn.utils import parametrize as parametrize

from orthogonium.layers.conv.AOC.fast_block_ortho_conv import (
    attach_bcop_weight,
)
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import conv_singular_values_numpy
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import fast_matrix_conv
from orthogonium.layers.conv.AOC.rko_conv import attach_rko_weight
from orthogonium.layers.conv.AOC.padding_convtranspose import PaddingConvTranspose2d
from orthogonium.reparametrizers import OrthoParams


class BcopRkoConv2d(nn.Conv2d):
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
        """
        This class handle native striding by combining a k-s x k-s convolution
        (without stride) with a sxs convolution (with stride s). This results in
        a kxk convolution with stride s. The first convolution is orthogonalized
        using BCOP and the second one is orthogonalized using RKO. By setting appropriate
        number of channels in the intermediate layer, the resulting convolution is
        orthogonal.

        It is advised to use the OrthogonalConv2d class instead of this one, as it
        instanciates the most efficient convolution for the given configuration.
        """
        super(BcopRkoConv2d, self).__init__(
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
        # raise runtime error if kernel size >= stride
        if self.kernel_size[0] < self.stride[0] or self.kernel_size[1] < self.stride[1]:
            raise ValueError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        if (
            ((max(in_channels, out_channels) // groups) < 2)
            and (self.kernel_size[0] != self.stride[0])
            and (self.kernel_size[1] != self.stride[1])
        ):
            raise ValueError("inner conv must have at least 2 channels")
        if (
            (self.out_channels >= self.in_channels)
            and (((self.dilation[0] % self.stride[0]) == 0) and (self.stride[0] > 1))
            and (((self.dilation[1] % self.stride[1]) == 0) and (self.stride[1] > 1))
        ):
            raise ValueError(
                "dilation must be 1 when stride is not 1. The set of orthonal convolutions is empty in this setting."
            )
        self.intermediate_channels = max(
            in_channels, out_channels // (self.stride[0] * self.stride[1])
        )
        del self.weight
        attach_bcop_weight(
            self,
            "weight_1",
            (
                self.intermediate_channels,
                in_channels // groups,
                max(1, self.kernel_size[0] - (self.stride[0] - 1)),
                max(1, self.kernel_size[1] - (self.stride[1] - 1)),
            ),
            groups,
            ortho_params,
        )
        attach_rko_weight(
            self,
            "weight_2",
            (
                out_channels,
                self.intermediate_channels // groups,
                self.stride[0],
                self.stride[1],
            ),
            groups,
            scale=1.0,
            ortho_params=ortho_params,
        )

    @property
    def weight(self):
        if self.training:
            return fast_matrix_conv(
                self.weight_1, self.weight_2, self.groups
            ).contiguous()
        else:
            with parametrize.cached():
                return fast_matrix_conv(
                    self.weight_1, self.weight_2, self.groups
                ).contiguous()


class BcopRkoConvTranspose2d(PaddingConvTranspose2d):
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
        """As BcopRkoConv2d handle native striding with explicit kernel. It unlocks
        the possibility to use the same parametrization for transposed convolutions.
        This class uses the same interface as the ConvTranspose2d class.

        Unfortunately, circular padding is not supported for the transposed convolution.
        But unit testing have shown that the convolution is still orthogonal when
         `out_channels * (self.stride[0]*self.stride[1]) > in_channels`.
        """
        super(BcopRkoConvTranspose2d, self).__init__(
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
        if (
            (self.out_channels <= self.in_channels)
            and (((self.dilation[0] % self.stride[0]) == 0) and (self.stride[0] > 1))
            and (((self.dilation[1] % self.stride[1]) == 0) and (self.stride[1] > 1))
        ):
            raise ValueError(
                "dilation must be 1 when stride is not 1. The set of orthonal convolutions is empty in this setting."
            )
        if self.kernel_size[0] < self.stride[0] or self.kernel_size[1] < self.stride[1]:
            raise ValueError(
                "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
            )
        if (
            ((max(in_channels, out_channels) // groups) < 2)
            and (self.kernel_size[0] != self.stride[0])
            and (self.kernel_size[1] != self.stride[1])
        ):
            raise ValueError("inner conv must have at least 2 channels")
        if out_channels * (self.stride[0] * self.stride[1]) >= in_channels:
            self.intermediate_channels = max(
                in_channels // (self.stride[0] * self.stride[1]), out_channels
            )
        else:
            self.intermediate_channels = out_channels
            # raise warning because this configuration don't yield orthogonal
            # convolutions
            # warnings.warn(
            #     "This configuration does not yield orthogonal convolutions due to "
            #     "padding issues: pytorch does not implement circular padding for "
            #     "transposed convolutions",
            #     RuntimeWarning,
            # )
        del self.weight
        attach_bcop_weight(
            self,
            "weight_1",
            (
                self.intermediate_channels,
                out_channels // groups,
                max(1, self.kernel_size[0] - (self.stride[0] - 1)),
                max(1, self.kernel_size[1] - (self.stride[1] - 1)),
            ),
            groups,
            ortho_params=ortho_params,
        )
        attach_rko_weight(
            self,
            "weight_2",
            (
                in_channels,
                self.intermediate_channels // groups,
                self.stride[0],
                self.stride[1],
            ),
            groups,
            scale=1.0,
            ortho_params=ortho_params,
        )

    @property
    def weight(self):
        if self.training:
            kernel = fast_matrix_conv(self.weight_1, self.weight_2, self.groups)
        else:
            with parametrize.cached():
                kernel = fast_matrix_conv(self.weight_1, self.weight_2, self.groups)
        return kernel.contiguous()
