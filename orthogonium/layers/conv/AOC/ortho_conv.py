from typing import Union

from torch import nn as nn
from torch.nn.common_types import _size_2_t

from orthogonium.layers.conv.AOC.bcop_x_rko_conv import BcopRkoConv2d
from orthogonium.layers.conv.AOC.bcop_x_rko_conv import BcopRkoConvTranspose2d
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import FastBlockConvTranspose2D
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import FastBlockConv2d
from orthogonium.layers.conv.AOC.rko_conv import RKOConv2d
from orthogonium.layers.conv.AOC.rko_conv import RkoConvTranspose2d
from orthogonium.reparametrizers import OrthoParams


def AdaptiveOrthoConv2d(
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
) -> nn.Conv2d:
    """
    Factory function to create an orthogonal convolutional layer, selecting the appropriate class based on kernel
    size and stride. This is the implementation for the `Adaptive Orthogonal Convolution` scheme [1]. It aims to be
    scalable to large networks and large image sizes, while enforcing orthogonality in the convolutional layers.
    This layer also intend to be compatible with all the feature of the `nn.Conv2d` class (e.g., striding, dilation,
    grouping, etc.). This method has an explicit kernel, which means that the forward operation is equivalent to a
    standard convolutional layer, but the weight are constrained to be orthogonal.

    Key Features:
    -------------
        - Enforces orthogonality, preserving gradient norms.
        - Supports native striding, dilation, grouped convolutions, and flexible padding.

    Behavior:
    ---------
        - When kernel_size == stride, the layer is an `RKOConv2d`.
        - When stride == 1, the layer is a `FastBlockConv2d`.
        - Otherwise, the layer is a `BcopRkoConv2d`.

    Note:
        - This implementation also work under zero padding, it lipschitz constant is still tight, but it looses
            orthogonality.orthogonality on the image border.
        - the unit tesing validated for a tolerance of 1e-4 under various orthogonalization schemes (see
            reparametrizers). Only Cholesky based methods were validated for a lower tolerance of 5e-2.

    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (_size_2_t): Size of the convolution kernel.
        stride (_size_2_t, optional): Stride of the convolution. Default is 1.
        padding (str or _size_2_t, optional): Padding mode or size. Default is "same".
        dilation (_size_2_t, optional): Dilation rate. Default is 1.
        groups (int, optional): Number of blocked connections from input to output channels. Default is 1.
        bias (bool, optional): Whether to include a learnable bias. Default is True.
        padding_mode (str, optional): Padding mode. Default is "circular".
        ortho_params (OrthoParams, optional): Parameters to control orthogonality. Default is `OrthoParams()`.

    Returns:
        A configured instance of `nn.Conv2d` (one of `RKOConv2d`, `FastBlockConv2d`, or `BcopRkoConv2d`).

    Raises:
        `ValueError`: If kernel_size < stride, as orthogonality cannot be enforced.


    References:
        - [1] Boissin, T., Mamalet, F., Fel, T., Picard, A. M., Massena, T., & Serrurier, M. (2025).
        An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures.
        <https://arxiv.org/abs/2501.07930>
    """

    if kernel_size < stride:
        raise ValueError(
            "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
        )
    if kernel_size == stride:
        convclass = RKOConv2d
    elif stride == 1:
        convclass = FastBlockConv2d
    else:
        convclass = BcopRkoConv2d
    return convclass(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        padding_mode=padding_mode,
        ortho_params=ortho_params,
    )


def AdaptiveOrthoConvTranspose2d(
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
) -> nn.ConvTranspose2d:
    """
    Factory function to create an orthogonal transposed convolutional layer, selecting the appropriate class based on kernel
    size and stride. This is the implementation for the `Adaptive Orthogonal Convolution` scheme [1]. It aims to be
    scalable to large networks and large image sizes, while enforcing orthogonality in the convolutional layers.
    This layer also intend to be compatible with all the feature of the `nn.Conv2d` class (e.g., striding, dilation,
    grouping, etc.). This method has an explicit kernel, which means that the forward operation is equivalent to a
    standard convolutional layer, but the weight are constrained to be orthogonal.

    Key Features:
    -------------
        - Ensures orthogonality in transpose convolutions for stable gradient propagation.
        - Supports dilation, grouped operations, and efficient kernel construction.

    Behavior:
    ---------
        - When kernel_size == stride, the layer is an `RkoConvTranspose2d`.
        - When stride == 1, the layer is a `FastBlockConvTranspose2D`.
        - Otherwise, the layer is a `BcopRkoConvTranspose2d`.


    Note:
        - This implementation also work under zero padding, it lipschitz constant is still tight, but it looses
            orthogonality.orthogonality on the image border.
        - The current implementation of the torch.nn.ConvTranspose2d does not support circular padding. One can
            implement padding manually by add a padding layer before and setting padding = (0,0).

    Arguments:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (_size_2_t): Size of the convolution kernel.
        stride (_size_2_t, optional): Stride of the transpose convolution. Default is 1.
        padding (_size_2_t, optional): Padding size. Default is 0.
        output_padding (_size_2_t, optional): Additional size for output. Default is 0.
        groups (int, optional): Number of groups. Default is 1.
        bias (bool, optional): Whether to include a learnable bias. Default is True.
        dilation (_size_2_t, optional): Dilation rate. Default is 1.
        padding_mode (str, optional): Padding mode. Default is "zeros".
        ortho_params (OrthoParams, optional): Parameters to control orthogonality. Default is `OrthoParams()`.

    Returns:
        A configured instance of `nn.ConvTranspose2d` (one of `RkoConvTranspose2d`, `FastBlockConvTranspose2D`, or `BcopRkoConvTranspose2d`).

    **Raises:**
    - `ValueError`: If kernel_size < stride, as orthogonality cannot be enforced.


    References:
        - [1] Boissin, T., Mamalet, F., Fel, T., Picard, A. M., Massena, T., & Serrurier, M. (2025).
        An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures.
        <https://arxiv.org/abs/2501.07930>
    """

    if kernel_size < stride:
        raise ValueError(
            "kernel size must be smaller than stride. The set of orthonal convolutions is empty in this setting."
        )
    if kernel_size == stride:
        convclass = RkoConvTranspose2d
    elif stride == 1:
        convclass = FastBlockConvTranspose2D
    else:
        convclass = BcopRkoConvTranspose2d
    return convclass(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=bias,
        dilation=dilation,
        padding_mode=padding_mode,
        ortho_params=ortho_params,
    )
