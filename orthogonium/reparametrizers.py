from dataclasses import dataclass
from typing import Callable
from typing import Tuple
import torch
import torch.nn.utils.parametrize as parametrize
from torch import nn as nn
from orthogonium.model_factory.classparam import ClassParam


class L2Normalize(nn.Module):
    def __init__(self, dtype, dim=None):
        """
        A class that performs L2 normalization for the given input tensor.

        L2 normalization is a process that normalizes the input over a specified
        dimension such that the sum of squares of the elements along that
        dimension equals 1. It ensures that the resulting tensor has a unit norm.
        This operation is widely used in machine learning and deep learning
        applications to standardize feature representations.

        Attributes:
            dim (Optional[int]): The specific dimension along which normalization
                is performed. If None, normalization is done over all dimensions.
            dtype (Any): The data type of the tensor to be normalized.

        Parameters:
            dtype: The data type of the tensor to be normalized.
            dim: An optional integer specifying the dimension along which to
                normalize. If not provided, the input will be normalized globally
                across all dimensions.
        """
        super(L2Normalize, self).__init__()
        self.dim = dim
        self.dtype = dtype

    def forward(self, x):
        return x / (torch.norm(x, dim=self.dim, keepdim=True, dtype=self.dtype) + 1e-8)

    def right_inverse(self, x):
        return x / (torch.norm(x, dim=self.dim, keepdim=True, dtype=self.dtype) + 1e-8)


# @torch.compile
class BatchedNewtonShultz(nn.Module):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. The code is derived from
    [the nano-GPT speedrun repo](https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py#L108) .
    With the coefficients from [@YouJiacheng](https://x.com/YouJiacheng/status/1893573513295897055/photo/1): these
    coefficients are different at each iteration, hence fixing the number of iterations. Original values implied
    6 iterations, but it seems that 4 iterations are enough to reach an orthogonal matrix with a good precision
    (sigma max = 1+-1e-2, tested with matrices up to 512x512).
    """

    def __init__(self, weight_shape):
        super(BatchedNewtonShultz, self).__init__()
        self.weight_shape = weight_shape

    # @torch.compile
    @staticmethod
    def zeropower_via_newtonschulz5(G: torch.Tensor) -> torch.Tensor:
        assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
        steps = 5  # number of iterations to perform
        # a, b, c =  (3.4445, -4.7750,  2.0315)#(2.0, 1.5, 0.5)
        coefs = [
            # (3.5,       -6126/1024,  3061/1024), # seems like the 2 first iteration can be removed without loss of precision
            # (3.5,       -7054/1024,  4050/1024),
            (3.5,       -8124/1024,  5372/1024),
            (3.5,       -9360/1024,  7130/1024),
            (3.5,       -7340/1024,  5451/1024),
            (3.5/1.75,  -2713/1024/1.75,  917/1024/1.75),
        ] # variable coefs for the quintic iteration, from https://x.com/YouJiacheng/status/1893573513295897055/photo/1
        # X = G.bfloat16()
        X = G
        if G.size(-2) > G.size(-1):
            X = X.mT

        # Ensure spectral norm is at most 1
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        # Perform the NS iterations
        for a, b, c in coefs:
            A = X @ X.mT
            B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X

        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

    def forward(self, w):
        return self.zeropower_via_newtonschulz5(w)

    def right_inverse(self, w):
        return w

class BatchedPowerIteration(nn.Module):
    def __init__(self, weight_shape, power_it_niter=3, eps=1e-12):
        """
        BatchedPowerIteration is a class that performs spectral normalization on weights
        using the power iteration method in a batched manner. It initializes singular
        vectors 'u' and 'v', which are used to approximate the largest singular value
        of the associated weight matrix during training. The L2 normalization is applied
        to stabilize these singular vector parameters.

        Attributes:
            weight_shape: tuple
                Shape of the weight tensor. Normalization is applied to the last two dimensions.
            power_it_niter: int
                Number of iterations to perform for the power iteration method.
            eps: float
                A small constant to ensure numerical stability during calculations. Used in the power iteration
                method to avoid dividing by zero.
        """
        super(BatchedPowerIteration, self).__init__()
        self.weight_shape = weight_shape
        self.power_it_niter = power_it_niter
        self.eps = eps
        # init u
        # u will be weight_shape[:-2] + (weight_shape[:-2], 1)
        # v will be weight_shape[:-2] + (weight_shape[:-1], 1,)
        self.u = nn.Parameter(
            torch.Tensor(torch.randn(*weight_shape[:-2], weight_shape[-2], 1)),
            requires_grad=False,
        )
        self.v = nn.Parameter(
            torch.Tensor(torch.randn(*weight_shape[:-2], weight_shape[-1], 1)),
            requires_grad=False,
        )
        parametrize.register_parametrization(
            self, "u", L2Normalize(dtype=self.u.dtype, dim=(-2))
        )
        parametrize.register_parametrization(
            self, "v", L2Normalize(dtype=self.v.dtype, dim=(-2))
        )

    def forward(self, X, init_u=None, n_iters=3, return_uv=True):
        if self.training:
            for _ in range(n_iters):
                self.v = X.transpose(-1, -2) @ self.u
                self.u = X @ self.v
        # stop gradient on u and v
        u = self.u.detach()
        v = self.v.detach()
        # but keep gradient on s
        s = u.transpose(-1, -2) @ X @ v
        return X / (s + self.eps)

    def right_inverse(self, normalized_kernel):
        # we assume that the kernel is normalized
        return normalized_kernel.to(self.u.dtype)


class BatchedIdentity(nn.Module):
    def __init__(self, weight_shape):
        """
        Class representing a batched identity matrix with a specific weight shape. The
        matrix is initialized based on the provided shape of the weights. It is a
        convenient utility for applications where identity-like operations are
        required in a batched manner.

        Attributes:
            weight_shape (Tuple[int, int]): A tuple representing the shape of the
            weight matrix for each batch. (unused)

        Args:
            weight_shape: A tuple specifying the shape of the individual weight matrix.
        """
        super(BatchedIdentity, self).__init__()

    def forward(self, w):
        return w

    def right_inverse(self, w):
        return w


class BatchedBjorckOrthogonalization(nn.Module):
    def __init__(self, weight_shape, beta=0.5, niters=12, pass_through=False):
        """
        Initialize the BatchedBjorckOrthogonalization module.

        This module implements the Björck orthogonalization method, which iteratively refines
        a weight matrix towards orthogonality. The method is especially effective when the
        weight matrix columns are nearly orthonormal. It balances computational efficiency
        with convergence speed through a user-defined `beta` parameter and iteration count.

        Args:
            weight_shape (tuple): The shape of the weight matrix to be orthogonalized.
            beta (float): Coefficient controlling the convergence of the orthogonalization process.
                Default is 0.5.
            niters (int): Number of iterations for the orthogonalization algorithm. Default is 12.
            pass_through (bool): If True, most iterations are performed without gradient computation,
                which can improve efficiency.
        """
        self.weight_shape = weight_shape
        self.beta = beta
        self.niters = niters
        self.pass_through = pass_through
        if weight_shape[-2] < weight_shape[-1]:
            self.wwtw_op = BatchedBjorckOrthogonalization.wwt_w_op
        else:
            self.wwtw_op = BatchedBjorckOrthogonalization.w_wtw_op
        super(BatchedBjorckOrthogonalization, self).__init__()

    @staticmethod
    def w_wtw_op(w):
        return w @ (w.transpose(-1, -2) @ w)

    @staticmethod
    def wwt_w_op(w):
        return (w @ w.transpose(-1, -2)) @ w

    def forward(self, w):
        """
        Apply the Björck orthogonalization process to the weight matrix.

        The algorithm adjusts the input matrix to approximate the closest orthogonal matrix
        by iteratively applying transformations based on the Björck algorithm.

        Args:
            w (torch.Tensor): The weight matrix to be orthogonalized.

        Returns:
            torch.Tensor: The orthogonalized weight matrix.
        """
        if self.pass_through:
            with torch.no_grad():
                for _ in range(self.niters):
                    w = (1 + self.beta) * w - self.beta * self.wwtw_op(w)
            # Final iteration without no_grad, using parameters:
            w = (1 + self.beta) * w - self.beta * self.wwtw_op(w)
        else:
            for _ in range(self.niters):
                w = (1 + self.beta) * w - self.beta * self.wwtw_op(w)
        return w

    def right_inverse(self, w):
        return w


class BatchedCholeskyOrthogonalization(nn.Module):
    def __init__(self, weight_shape, stable=False):
        """
        Initialize the BatchedCholeskyOrthogonalization module.

        This module orthogonalizes a weight matrix using the Cholesky decomposition method.
        It first computes the positive definite matrix \( V V^T \), then performs a Cholesky
        decomposition to obtain a lower triangular matrix. Solving the resulting triangular
        system yields an orthogonal matrix. This method is efficient and numerically stable,
        making it suitable for a wide range of applications.

        Args:
            weight_shape (tuple): The shape of the weight matrix.
            stable (bool): Whether to use the stable version of the Cholesky-based orthogonalization
                function, which adds a small positive diagonal element to ensure numerical stability.
                Default is False.
        """
        self.weight_shape = weight_shape
        super(BatchedCholeskyOrthogonalization, self).__init__()
        if stable:
            self.orth = BatchedCholeskyOrthogonalization.CholeskyOrthfn_stable.apply
        else:
            self.orth = BatchedCholeskyOrthogonalization.CholeskyOrthfn.apply

    # @staticmethod
    # def orth(X):
    #     S = X @ X.mT
    #     eps = S.diagonal(dim1=1, dim2=2).mean(1).mul(1e-3).detach()
    #     eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device)
    #     S = S + eps.view(-1, 1, 1) * eye.unsqueeze(0)
    #     L = torch.linalg.cholesky(S)
    #     W = torch.linalg.solve_triangular(L, X, upper=False)
    #     return W

    class CholeskyOrthfn(torch.autograd.Function):
        @staticmethod
        # def forward(ctx, X):
        #     S = X @ X.mT
        #     eps = S.diagonal(dim1=1, dim2=2).mean(1).mul(1e-3)
        #     eye = torch.eye(S.size(-1), dtype=S.dtype, device=S.device)
        #     S = S + eps.view(-1, 1, 1) * eye.unsqueeze(0)
        #     L = torch.linalg.cholesky(S)
        #     W = torch.linalg.solve_triangular(L, X, upper=False)
        #     ctx.save_for_backward(W, L)
        #     return W
        def forward(ctx, X):
            S = X @ X.mT
            eps = 1e-5  # A common stable choice
            S = S + eps * torch.eye(
                S.size(-1), dtype=S.dtype, device=S.device
            ).unsqueeze(0)
            L = torch.linalg.cholesky(S)
            W = torch.linalg.solve_triangular(L, X, upper=False)
            ctx.save_for_backward(W, L)
            return W

        @staticmethod
        def backward(ctx, grad_output):
            W, L = ctx.saved_tensors
            LmT = L.mT.contiguous()
            gB = torch.linalg.solve_triangular(LmT, grad_output, upper=True)
            gA = (-gB @ W.mT).tril()
            gS = (LmT @ gA).tril()
            gS = gS + gS.tril(-1).mT
            gS = torch.linalg.solve_triangular(LmT, gS, upper=True)
            gX = gS @ W + gB
            return gX

    class CholeskyOrthfn_stable(torch.autograd.Function):
        @staticmethod
        def forward(ctx, X):
            S = X @ X.mT
            eps = 1e-5  # A common stable choice
            S = S + eps * torch.eye(
                S.size(-1), dtype=S.dtype, device=S.device
            ).unsqueeze(0)
            L = torch.linalg.cholesky(S)
            W = torch.linalg.solve_triangular(L, X, upper=False)
            ctx.save_for_backward(X, W, L)
            return W

        @staticmethod
        def backward(ctx, grad_output):
            X, W, L = ctx.saved_tensors
            gB = torch.linalg.solve_triangular(L.mT, grad_output, upper=True)
            gA = (-gB @ W.mT).tril()
            gS = (L.mT @ gA).tril()
            gS = gS + gS.tril(-1).mT
            gS = torch.linalg.solve_triangular(L.mT, gS, upper=True)
            gS = torch.linalg.solve_triangular(L, gS, upper=False, left=False)
            gX = gS @ X + gB
            return gX

    def forward(self, w):
        """
        Apply Cholesky-based orthogonalization to the weight matrix.

        This method constructs a symmetric positive definite matrix from the input weight
        matrix, performs Cholesky decomposition, and solves the triangular system to produce
        an orthogonal matrix. It mimics the results of the Gram-Schmidt process but with
        improved numerical stability.

        Args:
            w (torch.Tensor): The weight matrix to be orthogonalized.

        Returns:
            torch.Tensor: The orthogonalized weight matrix.
        """
        return self.orth(w).view(*self.weight_shape)

    def right_inverse(self, w):
        return w


class BatchedExponentialOrthogonalization(nn.Module):
    def __init__(self, weight_shape, niters=7):
        """
        Initialize the BatchedExponentialOrthogonalization module.

        This module orthogonalizes a weight matrix using the exponential map of a skew-symmetric
        matrix. By converting the matrix into a skew-symmetric form and applying the matrix
        exponential, it produces an orthogonal matrix. This approach is particularly useful
        in contexts where smooth transitions between matrices are required.

        Non-square matrices are padded to the largest dimension to ensure that the matrix can
        be converted to a skew-symmetric matrix. The resulting matrix is cropped to the original
        dimension.

        Args:
            weight_shape (tuple): The shape of the weight matrix.
            niters (int): Number of iterations for the series expansion approximation of the
                matrix exponential. Default is 7.
        """
        self.weight_shape = weight_shape
        self.max_dim = max(weight_shape[-2:])
        self.niters = niters
        super(BatchedExponentialOrthogonalization, self).__init__()

    def forward(self, w):
        # fill w with zero to have a square matrix over the last two dimensions
        # if ((self.max_dim - w.shape[-1]) != 0) and ((self.max_dim - w.shape[-2]) != 0):
        w = torch.nn.functional.pad(
            w, (0, self.max_dim - w.shape[-1], 0, self.max_dim - w.shape[-2])
        )
        # makes w skew symmetric
        w = (w - w.transpose(-1, -2)) / 2
        acc = w
        res = torch.eye(acc.shape[-2], acc.shape[-1], device=w.device) + acc
        for i in range(2, self.niters):
            acc = torch.einsum("...ij,...jk->...ik", acc, w) / i
            res = res + acc
        # if transpose:
        #     res = res.transpose(-1, -2)
        res = res[..., : self.weight_shape[-2], : self.weight_shape[-1]]
        return res.contiguous()

    def right_inverse(self, w):
        return w


class BatchedQROrthogonalization(nn.Module):
    def __init__(self, weight_shape):
        """
        Initialize the BatchedQROrthogonalization module.

        This module uses QR decomposition to orthogonalize a weight matrix in a batched manner.
        It computes the orthogonal component (`Q`) from the decomposition, ensuring that the
        output satisfies orthogonality constraints.

        Args:
            weight_shape (tuple): The shape of the weight matrix to be orthogonalized.
        """
        super(BatchedQROrthogonalization, self).__init__()

    def forward(self, w):
        """
        Perform QR decomposition to compute the orthogonalized weight matrix.

        The QR decomposition splits the input matrix into an orthogonal matrix (`Q`) and
        an upper triangular matrix (`R`). This module returns the orthogonal component.

        Args:
            w (torch.Tensor): The weight matrix to be orthogonalized.

        Returns:
            torch.Tensor: The orthogonalized weight matrix (`Q` from the QR decomposition).
        """
        transpose = w.shape[-2] < w.shape[-1]
        if transpose:
            w = w.transpose(-1, -2)
        q, r = torch.linalg.qr(w, mode="reduced")
        # compute the sign of the diagonal of d
        diag_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
        # multiply the sign with the diagonal of r
        q = q * diag_sign
        if transpose:
            q = q.transpose(-1, -2)
        return q.contiguous()

    def right_inverse(self, w):
        return w


@dataclass
class OrthoParams:
    """
    Represents the parameters and configurations used for orthogonalization
    and spectral normalization.

    This class encapsulates the necessary modules and settings required
    for performing spectral normalization and orthogonalization of tensors
    in a parameterized way. It accommodates various implementations of
    normalizers and orthogonalization techniques to provide flexibility
    in their application. This way we can easily switch between different
    normalization techniques inside our layer despite that each normalization
    have different parameters.

    Attributes:
        spectral_normalizer (Callable[Tuple[int, ...], nn.Module]): A callable
            that produces a module for spectral normalization. Default is
            configured to use BatchedPowerIteration with specific parameters.
            This callable can be provided either as a `functool.partial` or as a
            `orthogonium.ClassParam`. It will recieve the shape of the weight tensor as its
            argument.
        orthogonalizer (Callable[Tuple[int, ...], nn.Module]): A callable
            that produces a module for orthogonalization. Default is
            configured to use BatchedBjorckOrthogonalization with specific
            parameters. This callable can be provided either as a `functool.partial` or as a
            `orthogonium.ClassParam`. It will recieve the shape of the weight tensor as its argument.
    """

    # spectral_normalizer: Callable[Tuple[int, ...], nn.Module] = BatchedIdentity
    spectral_normalizer: Callable[Tuple[int, ...], nn.Module] = ClassParam(  # type: ignore
        BatchedPowerIteration, power_it_niter=3, eps=1e-6
    )
    orthogonalizer: Callable[Tuple[int, ...], nn.Module] = ClassParam(  # type: ignore
        BatchedBjorckOrthogonalization,
        beta=0.5,
        niters=12,
        pass_through=False,
        # ClassParam(BatchedExponentialOrthogonalization, niters=12)
        # BatchedCholeskyOrthogonalization,
        # BatchedQROrthogonalization,
    )


DEFAULT_ORTHO_PARAMS = OrthoParams()
"""
The default orthogonalization parameters used by our library. 
Suitable for most applications and includes:
- A `BatchedPowerIteration` for spectral normalization
- A `BatchedBjorckOrthogonalization` for orthogonalization
"""

BJORCK_PASS_THROUGH_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=ClassParam(BatchedPowerIteration, power_it_niter=3, eps=1e-4),  # type: ignore
    orthogonalizer=ClassParam(
        BatchedBjorckOrthogonalization, beta=0.5, niters=12, pass_through=True
    ),
)
"""
Orthogonalization parameters that use the Bjorck orthogonalization method with
a pass-through optimization. This configuration greatly reduces the consumed memory but
at the cost of a slower convergence and worst perfomances.
"""

DEFAULT_TEST_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=ClassParam(BatchedPowerIteration, power_it_niter=4, eps=1e-4),  # type: ignore
    orthogonalizer=ClassParam(BatchedBjorckOrthogonalization, beta=0.5, niters=25),
    # orthogonalizer=ClassParam(BatchedQROrthogonalization),
    # orthogonalizer=ClassParam(BatchedExponentialOrthogonalization, niters=12),  # type: ignore
)
"""
Setting with more iterations to ensure that test passes with epsilon=1e-4.
"""

EXP_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=ClassParam(BatchedPowerIteration, power_it_niter=3, eps=1e-6),  # type: ignore
    orthogonalizer=ClassParam(BatchedExponentialOrthogonalization, niters=12),  # type: ignore
)
"""
Setting that use the exponential orthogonalization method with 12 iterations. The matrix is pre-conditionned
 with the power iteration method.
"""

QR_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=ClassParam(BatchedPowerIteration, power_it_niter=3, eps=1e-3),  # type: ignore
    orthogonalizer=ClassParam(BatchedQROrthogonalization),  # type: ignore
)
"""
Setting that use the QR orthogonalization method. The matrix is pre-conditionned
 with the power iteration method.
"""

CHOLESKY_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=BatchedIdentity,  # type: ignore
    orthogonalizer=ClassParam(BatchedCholeskyOrthogonalization),  # type: ignore
)
"""
Setting that use the Cholesky orthogonalization method. This method is memory and time efficient but
cannot converge to the exact orthogonal matrix (tests passing with epsilon=5e-5 meaning the layer may be 1.05 lipschitz). 
"""

CHOLESKY_STABLE_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=BatchedIdentity,
    orthogonalizer=ClassParam(BatchedCholeskyOrthogonalization, stable=True),
)
"""
Setting that use the Cholesky orthogonalization method and stores some values for backward to ensure numerical
 stability.
"""

NEWTONSHULTZ_ORTHO_PARAMS = OrthoParams(
    spectral_normalizer=BatchedIdentity,  # type: ignore
    orthogonalizer=ClassParam(BatchedNewtonShultz),
)
"""
Setting that use the Newton-Schultz orthogonalization method.
"""
