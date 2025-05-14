import torch
from torch import nn
from torch.nn.utils import parametrize

from orthogonium.layers.conv.AOC.fast_block_ortho_conv import (
    transpose_kernel,
    fast_matrix_conv,
)
from orthogonium.layers.conv.init import conv_orthogonal_


def safe_inv(x):
    mask = x == 0
    x_inv = x ** (-1)
    x_inv[mask] = 0
    return x_inv


class AOLReparametrizer(nn.Module):
    def __init__(self, nb_features, groups):
        super(AOLReparametrizer, self).__init__()
        self.nb_features = nb_features
        self.groups = groups
        self.q = nn.Parameter(torch.ones(nb_features, 1, 1, 1))

    def forward(self, kernel):
        ktk = fast_matrix_conv(
            transpose_kernel(kernel, self.groups, flip=True), kernel, self.groups
        )
        ktk = torch.abs(ktk)
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = (q_inv * ktk * q).sum((1, 2, 3))
        t = safe_inv(torch.sqrt(t))
        t = t.reshape(-1, 1, 1, 1)
        kernel = kernel * t
        return kernel


class MultiStepAOLReparametrizer(nn.Module):
    def __init__(self, nb_features, groups, niter=4):
        super(MultiStepAOLReparametrizer, self).__init__()
        self.groups = groups
        self.nb_features = nb_features
        self.niter = niter
        self.q = nn.Parameter(torch.ones(nb_features, 1, 1, 1))

    def forward(self, kernel):
        co, cig, ks, ks2 = kernel.shape
        if co // self.groups >= cig:
            kernel = transpose_kernel(kernel, self.groups, flip=True)
        kkt = kernel
        log_curr_norm = 0
        for i in range(self.niter):
            kkt_norm = kkt.norm().detach()
            kkt = kkt / kkt_norm
            log_curr_norm = 2 * (log_curr_norm + kkt_norm.log())
            kkt = fast_matrix_conv(
                transpose_kernel(kkt, self.groups, flip=True), kkt, self.groups
            )

        inverse_power = 2 ** (-self.niter)
        t = torch.abs(kkt)
        q = torch.exp(self.q)
        q_inv = torch.exp(-self.q)
        t = q_inv * t * q
        t = t.sum((1, 2, 3)).pow(inverse_power)
        norm = torch.exp(log_curr_norm * inverse_power)
        t = t * norm
        t = t.reshape(-1, 1, 1, 1)
        kernel = kernel / t
        if co // self.groups >= cig:
            kernel = transpose_kernel(kernel, self.groups, flip=True)
        return kernel

    def right_inverse(self, kernel):
        return kernel

    def reset_parameters(self):
        """
        Resets the parameters of the reparametrizer.
        """
        # Reset the q parameter to its initial value
        self.q.data.fill_(1.0)


class AOLConv2D(nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        niter=1,
    ):
        """
        Almost-Orthogonal Convolution layer. This layer implements the method proposed in [1] to enforce
        almost-orthogonality. While orthogonality is not enforced, the lipschitz constant of the layer
        is guaranteed to be less than 1.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolution kernel.
            stride (int or tuple, optional): Stride of the convolution. Default is 1.
            padding (int or tuple, optional): Padding size. Default is 0.
            dilation (int or tuple, optional): Dilation rate. Default is 1.
            groups (int, optional): Number of groups. Default is 1.
            bias (bool, optional): Whether to include a learnable bias. Default is True.
            padding_mode (str, optional): Padding mode. Default is "zeros".
            device (torch.device, optional): Device to store the layer parameters. Default is None.
            dtype (torch.dtype, optional): Data type to store the layer parameters. Default is None.


        References:
            `[1] Prach, B., & Lampert, C. H. (2022).
                   "Almost-orthogonal layers for efficient general-purpose lipschitz networks."
                   ECCV.`<https://arxiv.org/abs/2208.03160>`_
        """
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        self.niter = niter

        parametrize.register_parametrization(
            self,
            "weight",
            MultiStepAOLReparametrizer(
                min(out_channels, in_channels),
                groups=groups,
                niter=niter,
            ),
        )

    def reset_parameters(self) -> None:
        r"""Resets parameters of the module. This includes the weight and bias
        parameters, if they are used.
        """
        super().reset_parameters()
        # # Reset the parametrization
        # init kernel using the orthogonal kernel
        if not (
            self.in_channels // self.groups == 0
            and self.out_channels // self.groups == 0
        ):
            self.kernel = conv_orthogonal_(
                self.weight,
                stride=self.stride,
                groups=self.groups,
            )


class AOLConvTranspose2D(nn.ConvTranspose2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        niter=1,
    ):
        """
        Almost-Orthogonal Convolution layer. This layer implements the method proposed in [1] to enforce
        almost-orthogonality. While orthogonality is not enforced, the lipschitz constant of the layer
        is guaranteed to be less than 1.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int or tuple): Size of the convolution kernel.
            stride (int or tuple, optional): Stride of the convolution. Default is 1.
            padding (int or tuple, optional): Padding size. Default is 0.
            output_padding (int or tuple, optional): Additional size added to the output shape. Default is 0.
            groups (int, optional): Number of groups. Default is 1.
            bias (bool, optional): Whether to include a learnable bias. Default is True.
            dilation (int or tuple, optional): Dilation rate. Default is 1.
            padding_mode (str, optional): Padding mode. Default is "zeros".
            device (torch.device, optional): Device to store the layer parameters. Default is None.
            dtype (torch.dtype, optional): Data type to store the layer parameters. Default is None.


        References:
            `[1] Prach, B., & Lampert, C. H. (2022).
                   "Almost-orthogonal layers for efficient general-purpose lipschitz networks."
                   ECCV.`<https://arxiv.org/abs/2208.03160>`_
        """
        super().__init__(
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
            device=device,
            dtype=dtype,
        )
        self.niter = niter

        # Register the same AOLReparametrizer
        parametrize.register_parametrization(
            self,
            "weight",
            MultiStepAOLReparametrizer(
                min(out_channels, in_channels), groups=groups, niter=niter
            ),
        )
