import math
from typing import Optional, Sequence, Tuple

import torch
from orthogonium.layers.conv.AOC.fast_block_ortho_conv import (
    fast_matrix_conv,
    BCOPTrivializer,
)
from orthogonium.layers.conv.AOC.rko_conv import RKOParametrizer
from orthogonium.reparametrizers import OrthoParams, DEFAULT_ORTHO_PARAMS


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _pair(x: int | Sequence[int]) -> Tuple[int, int]:
    return (x, x) if isinstance(x, int) else tuple(x)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# orthogonal convolution weight initialiser
# ----------------------------------------------------------------------
def conv_orthogonal_(
    tensor: torch.Tensor,
    *,
    stride: int | Sequence[int] = 1,
    groups: int = 1,
    gain: float = 1.0,
    ortho_params: OrthoParams = DEFAULT_ORTHO_PARAMS,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    In‑place **orthogonal** initialisation of a 4‑D convolution weight.

    The routine mirrors the run‑time logic of BcopRkoConv2d:

    1. Build a BCOP kernel (weight₁) of size
       (intermediate_channels,  in_channels/groups, k', k'),
       where k' = kernel - (stride‑1).
    2. Build an RKO kernel (weight₂) with RKOParametrizer.
    3. Compose them with fast_matrix_conv → final (co, ci/groups, k, k)
       orthogonal kernel.
    4. Copy the result (scaled by *gain*) to **tensor**.

    Parameters
    ----------
    tensor : torch.Tensor
        Weight tensor of shape *(co, ci/groups, k, k)*.
    stride : int | tuple(int, int), default 1
        Stride that the convolution layer will use.
    groups : int, default 1
    gain : float, default 1.0
        Scaling factor (identical semantics to torch.nn.init.orthogonal_).
    ortho_params : OrthoParams, default DEFAULT_ORTHO_PARAMS
        Same factories used elsewhere in the code‑base.
    generator : torch.Generator | None, default None
        Optional RNG for reproducibility.

    Returns
    -------
    tensor : torch.Tensor
        The same tensor, now filled with an orthogonal kernel.
    """
    # ---------------- basic checks ------------------------------------------------
    if tensor.ndim != 4:
        raise ValueError("conv_orthogonal_ expects a 4‑D (co, ci, k, k) tensor")
    co, ci_per_group, k1, k2 = tensor.shape
    if k1 != k2:
        raise ValueError("only square kernels are supported")
    k = k1
    s1, s2 = _pair(stride)
    if k < s1 or k < s2:
        raise ValueError("kernel size must be ≥ stride in both spatial dims")

    device, dtype = tensor.device, tensor.dtype
    ci = ci_per_group * groups

    # ---------------- determine intermediate channels -----------------------------
    inter_channels = max(ci, co // (s1 * s2))
    inter_per_group = inter_channels // groups
    k_bcop = max(1, k - (s1 - 1))  # k' in the paper

    # ----------------------- 1) BCOP kernel  --------------------------------------
    num_proj = 2 * k_bcop
    pq_shape = (
        groups,
        num_proj,
        inter_per_group,  # max(ci, inter) // groups  == inter_per_group
        inter_per_group // 2,
    )
    PQ = torch.randn(pq_shape, device=device, dtype=dtype, generator=generator)
    with torch.no_grad():
        PQ = ortho_params.spectral_normalizer(weight_shape=pq_shape).to(device)(PQ)
        PQ = ortho_params.orthogonalizer(weight_shape=pq_shape).to(device)(PQ)

    trivialiser = BCOPTrivializer(
        in_channels=ci,
        out_channels=inter_channels,
        kernel_size=k_bcop,
        groups=groups,
    ).to(device, dtype)

    with torch.no_grad():
        weight1 = trivialiser(PQ)  # (inter_channels, ci/groups, k', k')

    # ----------------------- 2) RKO kernel  ---------------------------------------
    rko_shape = (co, inter_per_group, s1, s2)
    # raw parameter for the parametriser
    W2_raw = torch.randn(rko_shape, device=device, dtype=dtype, generator=generator)
    rko_param = RKOParametrizer(
        kernel_shape=rko_shape,
        groups=groups,
        scale=1.0,
        ortho_params=ortho_params,
    ).to(device, dtype)

    with torch.no_grad():
        weight2 = rko_param(W2_raw)  # orthogonal by construction

    # ----------------------- 3) compose & scale -----------------------------------
    kernel = fast_matrix_conv(weight1, weight2, groups)  # (co, ci/groups, k, k)

    with torch.no_grad():
        tensor.copy_(kernel.mul_(gain))

    return tensor
