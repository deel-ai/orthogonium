import warnings
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torch.nn.common_types import _size_2_t

from orthogonium.reparametrizers import L2Normalize
from orthogonium.reparametrizers import OrthoParams


def conv_singular_values_numpy(kernel, input_shape):
    """
    Hanie Sedghi, Vineet Gupta, and Philip M. Long. The singular values of convolutional layers.
    In International Conference on Learning Representations, 2019.
    """
    kernel = np.transpose(kernel, [0, 3, 4, 1, 2])  # g, k1, k2, ci, co
    transforms = np.fft.fft2(kernel, input_shape, axes=[1, 2])  # g, k1, k2, ci, co
    try:
        svs = np.linalg.svd(
            transforms, compute_uv=False, full_matrices=False
        )  # g, k1, k2, min(ci, co)
        stable_rank = (np.mean(svs) ** 2) / svs.max()
        return svs.min(), svs.max(), stable_rank
    except np.linalg.LinAlgError:
        print("numerical error in svd, returning only largest singular value")
        return None, np.linalg.norm(transforms, axis=(1, 2), ord=2), None


def fast_matrix_conv(m1, m2, groups=1):
    """Compute the convolution of two matrices using the block convolution operator.
    The original algorithm can be written as a single convolution operation, which
    unlock the massive parallelism of the convolution operator. This implementation
    is also more memory efficient than the original algorithm.

    Args:
        m1 (torch.Tensor): matrix of shape (c2, c1/g, k1, k2)
        m2 (torch.Tensor): matrix of shape (c3, c2/g, l1, l2)
        groups (int, optional): number of groups. Defaults to 1.

    Returns:
        torch.Tensor: result of the convolution of m1 and m2, this is a kernel of shape
        (c3, c1, k+l-1, k+l-1)

    """
    # m1 is m*n*k1*k2
    # m2 is nb*m*l1*l2
    m, n, k1, k2 = m1.shape
    nb, mb, l1, l2 = m2.shape
    assert m == mb * groups

    # Rearrange m1 for conv
    m1 = m1.transpose(0, 1)  # n*m*k1*k2

    # Rearrange m2 for conv
    m2 = m2.flip(-2, -1)

    # Run conv, output shape nb*n*(k+l-1)*(k+l-1)
    r2 = torch.nn.functional.conv2d(m1, m2, groups=groups, padding=(l1 - 1, l2 - 1))

    # Rearrange result
    return r2.transpose(0, 1)  # n*nb*(k+l-1)*(k+l-1)


def fast_batched_matrix_conv(m1, m2, groups=1):
    """Compute the convolution of two matrices using the block convolution operator.
    This is exactly the same as fast_matrix_conv but with an additional batch dimension.
    This is useful when we want to compute the convolution of multiple matrices in parallel.

    Args:
        m1 (torch.Tensor): matrix of shape (b, c2, c1/g, k1, k2)
        m2 (torch.Tensor): matrix of shape (b, c3, c2/g, l1, l2)
        groups (int, optional): number of groups. Defaults to 1.

    Returns:
        torch.Tensor: result of the convolution of m1 and m2, this is a kernel of shape
        (b, c3, c1, k+l-1, k+l-1)
    """
    b, m, n, k1, k2 = m1.shape
    b2, nb, mb, l1, l2 = m2.shape
    assert m == mb * groups
    assert b == b2
    m1 = m1.view(b * m, n, k1, k2)
    m2 = m2.view(b * nb, mb, l1, l2)
    # Rearrange m1 for conv
    m1 = m1.transpose(0, 1)  # n*m*k1*k2
    # Rearrange m2 for conv
    m2 = m2.flip(-2, -1)
    r2 = torch.nn.functional.conv2d(m1, m2, groups=groups * b, padding=(l1 - 1, l2 - 1))
    # Rearrange result
    r2 = r2.view(n, b, nb, k1 + l1 - 1, k2 + l2 - 1)
    r2 = r2.permute(1, 2, 0, 3, 4)
    return r2


def block_orth(p1, p2):
    """Construct a 2x2 orthogonal matrix from two orthogonal orthogonal projectors.
    Each projector can be seen as a 1x1 convolution, hence the stacking spatial stacking
    of [pi, I-pi] can be seen as a 2x1 or 1x2 orthogonal convolution. By using the block
    convolution operator, we can compute the 2x2 orthogonal conv. In this specific case,
    we can write the whole operation as a single einsum.

    Args:
        p1 (torch.Tensor): orthogonal projector of shape (g, x, c, c) where g is the number
            of groups, x is a batch dimension (allowing to compute the operation in parallel)
            and c is the number of channels.
        p2 (torch.Tensor): orthogonal projector of shape (g, x, c, c) also.

    Returns:
        torch.Tensor: orthogonal 2x2 conv of shape (x, g*c, c, 2, 2)
    """
    assert p1.shape == p2.shape
    g, x, n, n2 = p1.shape
    eye = torch.eye(n, device=p1.device, dtype=p1.dtype)
    # sorry for using x as a batch dimension, but this einsum was hard to write (thank you unit tests!)
    res = torch.einsum(
        "bgxij,cgxjk->xgikbc", torch.stack([p1, eye - p1]), torch.stack([p2, eye - p2])
    )
    # we reshape the result to get a 2x2 conv kernel
    res = res.reshape(x, g * n, n, 2, 2)
    return res


def transpose_kernel(p, groups, flip=True):
    """Compute the transpose of a kernel. This is done by transposing the kernel and
    flipping it along the last two dimensions. This operation is equivalent to the
    transpose of the convolution operator (when the stride is 1)

    Args:
        p (torch.Tensor): kernel of shape (cig, cog, k1, k2)
        groups (int): number of groups
        flip (bool, optional): if True, the kernel will be flipped. Defaults to True.
            False can be used when the is no need to flip the kernel.

    Returns:
        torch.Tensor: transposed kernel of shape (cog, cig, k1, k2)
    """
    cig, cog, k1, k2 = p.shape
    cig = cig // groups
    # we do not perform flip since it does not affect orthogonality
    p = p.view(groups, cig, cog, k1, k2)
    p = p.transpose(1, 2)
    if flip:
        p = p.flip(-1, -2)
    # merge groups to get the final kernel
    p = p.reshape(cog * groups, cig, k1, k2)
    return p


class BCOPTrivializer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        groups,
    ):
        """This module is used to generate orthogonal kernels for the BCOP layer. It takes
        as input a matrix PQ of shape (groups, 2*kernel_size, c, c//2) and returns a kernel
        of shape (c, c, kernel_size, kernel_size) that is orthogonal.

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of the kernel
            groups (int): number of groups
        """
        super(BCOPTrivializer, self).__init__()
        self.kernel_size = kernel_size
        self.groups = groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.min_channels = min(in_channels, out_channels)
        self.max_channels = max(in_channels, out_channels)
        self.transpose = out_channels < in_channels
        self.num_kernels = 2 * kernel_size

    def forward(self, PQ):
        ident = (
            torch.eye(self.max_channels // self.groups, device=PQ.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )
        # PQ contains 2*(kernel_size - 1) + 2 matrices of shape (c, c//2)
        # the 2 first matrices will be composed to build a (c, c) matrix
        # this (c, c) matrix will be used to build the first 1x1 conv
        # the remaining matrices will be used to build the 2x2 convs as
        # described in BCOP paper
        ####
        # first we compute PQ@PQ.t (used by both 1x1 and 2x2 convs)
        # we can rewrite PQ@PQ.t as an einsum
        PQ = torch.einsum("gijl,gikl->gijk", PQ, PQ)
        # PQ = PQ @ PQ.transpose(-1, -2)
        # we build the 1x1 conv using the two first matrices
        # we construct the (c, c) matrix by computing (I - 2*PQ[0]) @ (I - 2*PQ[1])
        # this is an extension of Householder reflection to the matrix case where
        # instead of reflecting a vector and compose c matrices, we reflect 2
        # (c, c//2) matrices. This results in an orthogonal matrix, but the fact that
        # any orthogonal matrix can be decomposed this way is yet to be proven.
        c11 = ident - 2 * PQ[:, 0]
        c11 = c11 @ (ident - 2 * PQ[:, 1])
        # reshape the matrix to build a 1x1 conv
        c11 = c11.view(
            self.max_channels,
            self.max_channels // self.groups,
            1,
            1,
        )
        # if the number of channels is different, we need to remove the extra channels
        # this results in a row/column othogonal matrix. It is still more efficient than
        # doing a separate orthogonalization (as shapes differs).
        if self.in_channels != self.out_channels:
            c11 = c11[:, : self.min_channels // self.groups, :, :]

        # build all 2x2 convs in parallel
        # half of the matrices will be used to create a 2x1 conv while the other half
        # will be used to create a 1x2 conv. The 2x1 and 1x2 convs will be composed
        # to build a 2x2 conv. c12 and c21 are notation abuse, since the tensors represent
        # 1x1 convs (it is the vertical/horizontal stacking of c12/c21 with (I-c12) and (I-c21)
        # that will result in a 1x2/2x1 conv)
        c12 = PQ[:, 2 : 2 + (self.kernel_size - 1), :, :]
        c21 = PQ[:, 2 + (self.kernel_size - 1) :, :, :]
        c22 = block_orth(
            c12, c21
        )  # this is an efficient and parallel way to compute the 2x2 convs
        # i used to belive that transposing half of the matrices would alleviate the expressiveness
        # issue, but it is not notable.
        # c22[1::2] = -c22[1::2].flip(-1, -2)

        # we now need to compose the 2x2 convs to build the k*k kernel
        # by using the associativity of the block conv operator we can
        # run the steps of the BCOP algorithm in parallel: we groups the
        # 2x2 convs in pairs and apply the block conv operator to each pair
        # until we have a single conv. If k-1 is a power of two this algorithm
        # run in log(k-1) steps. (naive associative scan algorithm)
        while (c22.shape[0] % 2 == 0) and (c22.shape[0] > 1):
            mid = c22.shape[0] // 2
            c22 = fast_batched_matrix_conv(c22[:mid], c22[mid:], self.groups)
        # we finally compose the 1x1 conv with the kxk conv
        res = c11
        for i in range(c22.shape[0]):  # c22.shape[0] == 1 if k-1 is a power of two
            res = fast_matrix_conv(res, c22[i], self.groups)
        # since it is less expensive to compute the transposed kernel when co < ci
        # we transpose the kernel if needed
        if self.transpose:
            res = transpose_kernel(res, self.groups, flip=False)
        # it seems more efficient to make the kernel contiguous since it will be used
        # in a convolution
        return res.contiguous()


def attach_bcop_weight(
    layer, weight_name, kernel_shape, groups, ortho_params: OrthoParams = OrthoParams()
):
    """
    Attach a weight to a layer and parametrize it with the BCOPTrivializer module.
    The attached weight will be the kernel of an orthogonal convolutional layer.

    Args:
        layer (torch.nn.Module): layer to which the weight will be attached
        weight_name (str): name of the weight
        kernel_shape (tuple): shape of the kernel (out_channels, in_channels/groups, kernel_size, kernel_size)
        groups (int): number of groups
        bjorck_params (BjorckParams, optional): parameters of the Bjorck orthogonalization. Defaults to BjorckParams().

    Returns:
        torch.Tensor: a handle to the attached weight
    """
    out_channels, in_channels, kernel_size, k2 = kernel_shape
    in_channels *= groups  # compute the real number of input channels
    assert kernel_size == k2, "only square kernels are supported for the moment"
    max_channels = max(in_channels, out_channels)
    num_kernels = (
        2 * kernel_size
    )  # the number of projectors needed to create the kernel
    # register projectors matrices
    layer.register_parameter(
        weight_name,
        torch.nn.Parameter(
            torch.Tensor(
                groups,
                num_kernels,
                (max_channels // groups),
                (max_channels // (groups * 2)),
            ),
            requires_grad=True,
        ),
    )
    weight = getattr(layer, weight_name)
    torch.nn.init.orthogonal_(weight)
    if weight.shape[-1] == 1:
        # if max_channels//groups == 1, we can use L2 normalization
        # instead of Bjorck orthogonalization which is significantly faster
        parametrize.register_parametrization(
            layer,
            weight_name,
            L2Normalize(dtype=weight.dtype, dim=(-2)),
        )
    else:
        # register power iteration and Bjorck orthogonalization
        parametrize.register_parametrization(
            layer,
            weight_name,
            ortho_params.spectral_normalizer(weight_shape=weight.shape),
        )
        parametrize.register_parametrization(
            layer,
            weight_name,
            ortho_params.orthogonalizer(
                weight_shape=weight.shape,
            ),
        )
    # now we have orthogonal projectors, we can build the orthogonal kernel
    parametrize.register_parametrization(
        layer,
        weight_name,
        BCOPTrivializer(
            in_channels,
            out_channels,
            kernel_size,
            groups,
        ),
        unsafe=True,
    )
    return weight


class FastBlockConv2d(nn.Conv2d):
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
        Fast implementation of the Block Circulant Orthogonal Parametrization (BCOP) for convolutional layers.
        This approach changes the original BCOP algorithm to make it more scalable and efficient. This implementation
        rewrite efficiently the block convolution operator as a single convolution operation. Also the iterative algorithm
        is parallelized in the associative scan fashion.

        This layer is a drop-in replacement for the nn.Conv2d layer. It is orthogonal and Lipschitz continuous while maintaining
        the same interface as the Con2d. Also this method has an explicit kernel, whihc allows to compute the singular values of
        the convolutional layer.

        Striding is not supported when out_channels > in_channels. Real striding is supported in BcopRkoConv2d. The use of
        OrthogonalConv2d is recommended.
        """
        super(FastBlockConv2d, self).__init__(
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
            (self.stride[0] > 1 or self.stride[1] > 1) and (out_channels > in_channels)
        ) or (
            self.stride[0] > self.kernel_size[0] or self.stride[1] > self.kernel_size[1]
        ):
            raise ValueError(
                "stride > 1 is not supported when out_channels > in_channels, "
                "use TODO layer instead"
            )
        if (
            (self.out_channels >= self.in_channels)
            and (((self.dilation[0] % self.stride[0]) == 0) and (self.stride[0] > 1))
            and (((self.dilation[1] % self.stride[1]) == 0) and (self.stride[1] > 1))
        ):
            raise ValueError(
                "dilation must be 1 when stride is not 1. The set of orthonal convolutions is empty in this setting."
            )
        del self.weight
        attach_bcop_weight(
            self,
            "weight",
            (
                out_channels,
                in_channels // groups,
                self.kernel_size[0],
                self.kernel_size[1],
            ),
            groups,
            ortho_params=ortho_params,
        )


class FastBlockConvTranspose2D(nn.ConvTranspose2d):
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
        Extention of the BCOP algorithm to transposed convolutions. This implementation
        uses the same algorithm as the FlashBCOP layer, but the layer acts as a transposed
        convolutional layer.
        """
        super(FastBlockConvTranspose2D, self).__init__(
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
        del self.weight
        attach_bcop_weight(
            self,
            "weight",
            (
                in_channels,
                out_channels // self.groups,
                self.kernel_size[0],
                self.kernel_size[1],
            ),
            groups,
            ortho_params=ortho_params,
        )
