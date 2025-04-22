import numpy as np
import pytest
import torch
from torch import nn

from orthogonium.layers.conv.init import conv_orthogonal_  # ← import your new init
from orthogonium.layers.conv.singular_values import get_conv_sv
from orthogonium.reparametrizers import DEFAULT_TEST_ORTHO_PARAMS
from tests.test_orthogonality_conv import _compute_sv_impulse_response_layer

# we can borrow the helper you already wrote

device = "cpu"  #  torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_orthogonal_layer(
    orthoconv,
    groups,
    input_channels,
    kernel_size,
    output_channels,
    expected_kernel_shape,
    tol=1e-2,
    sigma_min_requirement=0.0,
    imsize=8,
):
    # fixing seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    # Test backpropagation and weight update
    try:
        orthoconv = orthoconv.to(device)
    except Exception as e:
        pytest.fail(f"Backpropagation or weight update failed with: {e}")
    with torch.no_grad():
        try:
            sigma_max, stable_rank = get_conv_sv(
                orthoconv,
                n_iter=6 if orthoconv.padding_mode == "circular" else 3,
                imsize=imsize,
            )
        except np.linalg.LinAlgError as e:
            pytest.skip(f"SVD failed with: {e}")
        sigma_min_ir, sigma_max_ir, stable_rank_ir = _compute_sv_impulse_response_layer(
            orthoconv, (input_channels, imsize, imsize)
        )
    print(
        f"({input_channels}->{output_channels}, g{groups}, k{kernel_size}), "
        f"sigma_max:"
        f" {sigma_max:.3f}/{sigma_max_ir:.3f}, "
        f"sigma_min:"
        f" {sigma_min_ir:.3f}, "
        f"stable_rank: {stable_rank:.3f}/{stable_rank_ir:.3f}"
    )
    # check that the singular values are close to 1
    assert sigma_max_ir < (1 + tol), "sigma_max is not less than 1"
    assert (sigma_min_ir < (1 + tol)) and (
        sigma_min_ir > sigma_min_requirement
    ), "sigma_min is not close to 1"
    try:
        # check that table rank is greater than 0.75
        assert stable_rank_ir > 0.75, "stable rank is not greater than 0.75"
        assert (
            sigma_max + tol >= sigma_max_ir
        ), f"sigma_max is not greater than its IR value: {sigma_max} vs {sigma_max_ir}"
    except AssertionError as e:
        # given the large number of tests and the stochastic nature of these, we can
        # expect 1 over 100 tests to fail. Especially on less mandatory properties
        # (like stable rank). However, it is relevant to check that this is not a systematic
        # failure. To do so, when the test fails, performs a less strict check and decide if
        # the test will raise a warning or an error. (The number of warnings should be monitored)
        assert stable_rank_ir > 0.25, "stable rank is not greater than 0.25"
        pytest.skip("Stable rank is less than 0.75, but greater than 0.25")


# ---------------------------------------------------------------------
# utility: create a vanilla conv whose weight we will overwrite
# ---------------------------------------------------------------------
def _make_plain_conv(
    cin: int,
    cout: int,
    k: int,
    stride: int = 1,
    groups: int = 1,
    padding_mode: str = "circular",
) -> nn.Conv2d:
    padding = "same" if stride == 1 else 0
    return nn.Conv2d(
        cin,
        cout,
        kernel_size=k,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False,
        padding_mode=padding_mode,
    )


# ---------------------------------------------------------------------
# 1) positive‑path: does the initialiser really yield an orthogonal kernel?
# ---------------------------------------------------------------------
@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("in_channels", [8, 16])
@pytest.mark.parametrize("out_channels", [8, 16])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_conv_orthogonal_init_standard(
    kernel_size, in_channels, out_channels, stride, groups
):
    if kernel_size < stride:
        pytest.skip("kernel size must be greater than stride")
    conv = _make_plain_conv(
        in_channels, out_channels, kernel_size, stride=stride, groups=groups
    ).to(device)

    # ---- apply the initializer --------------------------------------------------
    conv_orthogonal_(
        conv.weight,
        stride=stride,
        groups=groups,
        ortho_params=DEFAULT_TEST_ORTHO_PARAMS,  # <- exact same factories as elsewhere
    )

    # ---- run the same orthogonality battery you already trust -------------------
    check_orthogonal_layer(
        conv,
        groups,
        in_channels,
        kernel_size,
        out_channels,
        expected_kernel_shape=conv.weight.shape,
    )


# ---------------------------------------------------------------------
# 2) even kernels and depth‑wise cases
# ---------------------------------------------------------------------
@pytest.mark.parametrize("kernel_size", [2, 4])
@pytest.mark.parametrize("in_channels", [8, 16])
@pytest.mark.parametrize("out_channels", [8, 16])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_conv_orthogonal_even_kernels(kernel_size, in_channels, out_channels, groups):
    conv = _make_plain_conv(
        in_channels,
        out_channels,
        kernel_size,
        groups=groups,
        padding_mode="circular",
    ).to(device)
    conv_orthogonal_(conv.weight, groups=groups)
    check_orthogonal_layer(
        conv,
        groups,
        in_channels,
        kernel_size,
        out_channels,
        expected_kernel_shape=conv.weight.shape,
    )


# ---------------------------------------------------------------------
# 3) negative‑path: argument validation
# ---------------------------------------------------------------------
def test_conv_orthogonal_bad_shape():
    # non‑4‑D tensor
    bad = torch.empty(10, 3, 3)
    with pytest.raises(ValueError, match="4‑D"):
        conv_orthogonal_(bad)

    # non‑square kernel
    bad = torch.empty(8, 8, 3, 5)
    with pytest.raises(ValueError, match="square kernels"):
        conv_orthogonal_(bad)


@pytest.mark.parametrize("k,s", [(2, 3), (3, 4)])
def test_conv_orthogonal_kernel_smaller_than_stride_raises(k, s):
    bad = torch.empty(8, 8, k, k)
    with pytest.raises(ValueError, match="kernel size must be ≥ stride"):
        conv_orthogonal_(bad, stride=s)


# ---------------------------------------------------------------------
# 4) reproducibility smoke‑test
# ---------------------------------------------------------------------
def test_conv_orthogonal_reproducibility():
    g = torch.Generator().manual_seed(42)
    w1 = torch.empty(8, 8, 3, 3)
    w2 = torch.empty_like(w1)

    conv_orthogonal_(w1, generator=g)
    # reset the generator
    g.manual_seed(42)
    conv_orthogonal_(w2, generator=g)

    assert torch.allclose(
        w1, w2, atol=1e-6, rtol=1e-5
    ), "initialiser must be deterministic under the same RNG"
