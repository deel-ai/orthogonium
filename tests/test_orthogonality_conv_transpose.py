import numpy as np
import pytest
import torch
from orthogonium.layers.conv.AOC.ortho_conv import AdaptiveOrthoConvTranspose2d
from tests.test_orthogonality_conv import check_orthogonal_layer
from orthogonium.layers.conv.AOC import (
    BcopRkoConvTranspose2d,
    FastBlockConvTranspose2D,
    RkoConvTranspose2d,
)
from orthogonium.reparametrizers import (
    DEFAULT_TEST_ORTHO_PARAMS,
    EXP_ORTHO_PARAMS,
    CHOLESKY_ORTHO_PARAMS,
    QR_ORTHO_PARAMS,
    CHOLESKY_STABLE_ORTHO_PARAMS,
)


@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("input_channels", [4, 8, 16])
@pytest.mark.parametrize("output_channels", [4, 8, 32])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2])
def test_convtranspose(kernel_size, input_channels, output_channels, stride, groups):
    # Test instantiation
    padding = (0, 0)
    padding_mode = "zeros"
    try:

        orthoconvtranspose = AdaptiveOrthoConvTranspose2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=padding,
            padding_mode=padding_mode,
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    if (
        kernel_size > 1
        and kernel_size != stride
        and output_channels * (stride**2) < input_channels
    ):
        pytest.skip("this case is not handled yet")
    check_orthogonal_layer(
        orthoconvtranspose,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            input_channels,
            output_channels // groups,
            kernel_size,
            kernel_size,
        ),
    )


def test_invalid_kernel_smaller_than_stride():
    """
    A test to ensure that kernel_size < stride raises an expected ValueError
    """
    with pytest.raises(ValueError, match=r"kernel size must be smaller than stride"):
        AdaptiveOrthoConvTranspose2d(
            in_channels=8,
            out_channels=16,
            kernel_size=2,
            stride=3,  # Invalid: kernel_size < stride
            groups=1,
        )
    with pytest.raises(ValueError, match=r"kernel size must be smaller than stride"):
        BcopRkoConvTranspose2d(
            in_channels=8,
            out_channels=16,
            kernel_size=2,
            stride=3,  # Invalid: kernel_size < stride
            groups=1,
        )
    with pytest.raises(ValueError, match=r"kernel size must be smaller than stride"):
        FastBlockConvTranspose2D(
            in_channels=8,
            out_channels=16,
            kernel_size=2,
            stride=3,  # Invalid: kernel_size < stride
            groups=1,
        )


@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("input_channels", [4, 8])
@pytest.mark.parametrize("output_channels", [4, 8])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize(
    "ortho_params",
    [
        "default_bb",
        "exp",
        "qr",
        "cholesky",
        "cholesky_stable",
    ],
)
def test_parametrizers_standard_configs(
    kernel_size, input_channels, output_channels, stride, groups, ortho_params
):
    """
    test combinations of kernel size, input channels, output channels, stride and groups
    """
    ortho_params_dict = {
        "default_bb": DEFAULT_TEST_ORTHO_PARAMS,
        "exp": EXP_ORTHO_PARAMS,
        "qr": QR_ORTHO_PARAMS,
        "cholesky": CHOLESKY_ORTHO_PARAMS,
        "cholesky_stable": CHOLESKY_STABLE_ORTHO_PARAMS,
    }  # trick to have the actual method name displayed properly if test fails
    # Test instantiation
    padding = (0, 0)
    padding_mode = "zeros"
    try:
        orthoconvtranspose = AdaptiveOrthoConvTranspose2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=padding,
            padding_mode=padding_mode,
            ortho_params=ortho_params_dict[ortho_params],
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    if (
        kernel_size > 1
        and kernel_size != stride
        and output_channels * (stride**2) < input_channels
    ):
        pytest.skip("this case is not handled yet")
    check_orthogonal_layer(
        orthoconvtranspose,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            input_channels,
            output_channels // groups,
            kernel_size,
            kernel_size,
        ),
        tol=5e-2 if ortho_params.startswith("cholesky") else 1e-3,
        sigma_min_requirement=0.75 if ortho_params.startswith("cholesky") else 0.95,
    )



@pytest.mark.parametrize("kernel_size", [1, 2, 3])
@pytest.mark.parametrize("input_channels", [4, 8, 16])
@pytest.mark.parametrize("output_channels", [4, 8, 32])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2])
def test_convtranspose_circular(kernel_size, input_channels, output_channels, stride, groups):
    # Test instantiation
    padding = "same"
    padding_mode = "circular"
    try:

        orthoconvtranspose = AdaptiveOrthoConvTranspose2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=padding,
            padding_mode=padding_mode,
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    if (
        kernel_size > 1
        and kernel_size != stride
        and output_channels * (stride**2) < input_channels
    ):
        pytest.skip("this case is not handled yet")
    check_orthogonal_layer(
        orthoconvtranspose,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            input_channels,
            output_channels // groups,
            kernel_size,
            kernel_size,
        ),
        check_same_padding=True,
        stride = stride,
    )
