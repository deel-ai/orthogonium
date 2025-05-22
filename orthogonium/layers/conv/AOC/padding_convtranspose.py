from typing import Union

import math

from torch import nn as nn
from torch.nn.common_types import _size_2_t

from orthogonium.reparametrizers import OrthoParams


class PaddingConvTranspose2d(nn.ConvTranspose2d):
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
        """
        Wrapper for circular padding on ConvTransposed2d
        padding_mode can be "zeros" or "circular"
        "zeros" is the default padding mode for ConvTranspose2d and can only orthogonal with padding = 0
        "circular" option is only implemented and orthogonal for a padding="same"
        """
        if (kernel_size == 1) or (kernel_size == stride): # rko
            padding = 0
            padding_mode = "zeros"
            print(f"zero padding valid kernel size = {kernel_size} stride = {stride}")
        self.pad4transpose = None
        if padding_mode == "circular":
            if padding !=  "same":
                raise ValueError(
                    "circular padding is only supported with padding = \"same\".")
            if stride == 1:
                k1 = k2 = kernel_size
            else:
                k1 = 1+ 2*((kernel_size-1)//(2*stride)) # first sub kernel size : center and twice 1/s pixels on each side
                k2 = math.ceil((kernel_size - k1)/(stride-1)) # max of other sub kernel size
            padd_circ = (k1//2, k2//2, k1//2, k2//2)
            self.pad4transpose = tuple(padd_circ)
            self.crop_left = stride*(k1//2)
            self.crop_right = -stride*(k2//2 - 1) - kernel_size%2
            if self.crop_right == 0:
                self.crop_right = None
            print(self.pad4transpose, self.crop_left, self.crop_right)
            padding = kernel_size//2
            padding_mode = "zeros"

        super(PaddingConvTranspose2d, self).__init__(
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

    def forward(self, input):
        if self.pad4transpose is not None:
            input = nn.functional.pad(
                input, self.pad4transpose, mode="circular"
            )
        output = super(PaddingConvTranspose2d, self).forward(input)
        if self.pad4transpose is not None:
            output = output[:, :, self.crop_left:self.crop_right, self.crop_left:self.crop_right]
        return output.contiguous()
