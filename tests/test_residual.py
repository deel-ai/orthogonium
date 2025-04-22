# test_residuals.py

import pytest
import torch
from torch import nn

from orthogonium.layers.residual import (
    ConcatResidual,
    L2NormResidual,
    AdditiveResidual,
    PrescaledAdditiveResidual,
)
from orthogonium.layers.conv.AOC import AdaptiveOrthoConv2d


def compute_lipschitz_constant(layer: nn.Module, x: torch.Tensor) -> float:
    """
    Compute the operator norm (L2→2) of `layer` at point `x`
    by building the full Jacobian and taking its spectral norm.
    """
    x = x.clone().detach().requires_grad_(True)
    y = layer(x)

    # build Jacobian matrix J where J[i, j] = dy_i / dx_j
    jac_rows = []
    flat_x = x.view(-1)
    for i in range(y.numel()):
        grad_out = torch.zeros_like(y)
        grad_out.view(-1)[i] = 1.0
        (grad_x,) = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_out,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )
        jac_rows.append(grad_x.view(-1))
    J = torch.stack(jac_rows, dim=0)  # shape (y.numel(), x.numel())

    # spectral norm = operator 2-norm = largest singular value
    return torch.linalg.matrix_norm(J, ord=2).item()


@pytest.mark.parametrize(
    "residual_cls, fn_kwargs, x_shape",
    [
        # 👇 batch=1, 2×4=8 channels for ConcatResidual
        (
            ConcatResidual,
            {"in_channels": 4, "out_channels": 4, "kernel_size": 3},
            (1, 8, 8, 8),
        ),
        # 👇 batch=1, 4 channels for the other three
        (
            L2NormResidual,
            {"in_channels": 4, "out_channels": 4, "kernel_size": 3},
            (1, 4, 8, 8),
        ),
        (
            AdditiveResidual,
            {"in_channels": 4, "out_channels": 4, "kernel_size": 3},
            (1, 4, 8, 8),
        ),
        (
            PrescaledAdditiveResidual,
            {"in_channels": 4, "out_channels": 4, "kernel_size": 3},
            (1, 4, 8, 8),
        ),
    ],
)
def test_shape_grad_and_lipschitz(residual_cls, fn_kwargs, x_shape):
    # instantiate inner block and wrap
    fn_block = AdaptiveOrthoConv2d(**fn_kwargs)
    layer = residual_cls(fn_block)

    # random input
    x = torch.randn(*x_shape, requires_grad=True)

    # — shape check —
    out = layer(x)
    assert out.shape == x.shape, (
        f"{residual_cls.__name__} changed shape: "
        f"got {out.shape}, expected {x_shape}"
    )

    # — Lipschitz constant check at init —
    lip = compute_lipschitz_constant(layer, x)
    print(f"{residual_cls.__name__} | Lipschitz constant: {lip:.6f}")
    assert (
        lip <= 1.0 + 1e-4
    ), f"{residual_cls.__name__} violates Lipschitz ≤1: got {lip:.6f}"

    # — gradient‐flow check —
    out.sum().backward()
    assert x.grad is not None, "No gradient on the input!"

    # at least one inner param saw a grad
    inner_grads = [p.grad is not None for p in layer.fn.parameters()]
    assert any(inner_grads), "No gradients in any inner block parameters!"

    # wrapper alpha, if it exists
    if hasattr(layer, "alpha"):
        assert (
            layer.alpha.grad is not None
        ), f"No gradient on alpha of {residual_cls.__name__}"
