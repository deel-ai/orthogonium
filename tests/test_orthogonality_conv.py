import numpy as np
import pytest
import torch
from orthogonium.layers import AdaptiveOrthoConv2d
from orthogonium.layers.conv.AOC import BcopRkoConv2d, FastBlockConv2d, RKOConv2d
from orthogonium.layers.conv.singular_values import get_conv_sv
from orthogonium.reparametrizers import (
    DEFAULT_TEST_ORTHO_PARAMS,
    EXP_ORTHO_PARAMS,
    CHOLESKY_ORTHO_PARAMS,
    QR_ORTHO_PARAMS,
    CHOLESKY_STABLE_ORTHO_PARAMS,
    NEWTONSHULTZ_ORTHO_PARAMS,
)

device = "cpu"  #  torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_sv_impulse_response_layer(layer, img_shape):
    # fixing seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    with torch.no_grad():
        layer = layer.to(device)
        inputs = (
            torch.eye(img_shape[0] * img_shape[1] * img_shape[2])
            .view(
                img_shape[0] * img_shape[1] * img_shape[2],
                img_shape[0],
                img_shape[1],
                img_shape[2],
            )
            .to(device)
        )
        outputs = layer(inputs)
        try:
            outputs_reshaped = outputs.view(outputs.shape[0], -1)
            sv_max = torch.linalg.norm(outputs_reshaped, ord=2)
            sv_min = torch.linalg.norm(outputs_reshaped, ord=-2)
            fro_norm = torch.linalg.norm(outputs_reshaped, ord="fro")
            # svs = torch.linalg.svdvals(outputs.view(outputs.shape[0], -1))
            # svs = svs.cpu()
            # return svs.min(), svs.max(), svs.mean() / svs.max()
            return (
                sv_min,
                sv_max,
                fro_norm**2
                / (
                    sv_max**2
                    * min(outputs_reshaped.shape[0], outputs_reshaped.shape[1])
                ),
            )
        except np.linalg.LinAlgError:
            print("SVD failed returning only largest singular value")
            return torch.norm(outputs.view(outputs.shape[0], -1), p=2).max(), 0, 0


def check_orthogonal_layer(
    orthoconv,
    groups,
    input_channels,
    kernel_size,
    output_channels,
    expected_kernel_shape,
    tol=5e-4,
    sigma_min_requirement=0.95,
    check_same_padding= False,
    stride = 1
):
    # fixing seeds for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    imsize = 8
    # Test backpropagation and weight update
    try:
        orthoconv = orthoconv.to(device)
        orthoconv.train()
        opt = torch.optim.SGD(orthoconv.parameters(), lr=0.001)
        for i in range(25):
            opt.zero_grad()
            inp = torch.randn(1, input_channels, imsize, imsize).to(device)
            output = orthoconv(inp)
            loss = -output.mean()
            loss.backward()
            opt.step()
        orthoconv.eval()  # so i    mpulse response test checks the eval mode
    except Exception as e:
        pytest.fail(f"Backpropagation or weight update failed with: {e}")
    # check that orthoconv.weight has the correct shape
    if orthoconv.weight.data.shape != expected_kernel_shape:
        pytest.fail(
            f"BCOP weight has incorrect shape: {orthoconv.weight.shape} vs {(output_channels, input_channels // groups, kernel_size, kernel_size)}"
        )
    # check that the output shape is correct
    if check_same_padding:
        if output.shape[1:] != (output_channels, imsize*stride, imsize*stride):
            pytest.fail(
                f"BCOP output has incorrect shape: {output.shape[1:]} vs {(output_channels, imsize*stride, imsize*stride)}"
            )

    # Test singular_values function
    sigma_max, stable_rank = get_conv_sv(
        orthoconv,
        n_iter=6 if orthoconv.padding_mode == "circular" else 3,
        imsize=imsize,
    )
    sigma_min_ir, sigma_max_ir, stable_rank_ir = _compute_sv_impulse_response_layer(
        orthoconv, (input_channels, imsize, imsize)
    )
    print(f"input_shape = {inp.shape}, output_shape = {output.shape}")
    print(
        f"({input_channels}->{output_channels}, g{groups}, k{kernel_size}), "
        f"sigma_max:"
        f" {sigma_max:.3f}/{sigma_max_ir:.3f}(ir), "
        f"sigma_min:"
        f" {sigma_min_ir:.3f}, "
        f"stable_rank: {stable_rank:.3f}/{stable_rank_ir:.3f}(ir)"
    )
    # check that the singular values are close to 1
    assert sigma_max_ir < (1 + tol), "sigma_max is not less than 1"
    assert (sigma_min_ir < (1 + tol)) and (
        sigma_min_ir > sigma_min_requirement
    ), "sigma_min is not close to 1"
    assert abs(stable_rank_ir - 1) < tol, "stable_rank is not close to 1"
    # check that the singular values are close to the impulse response values
    # assert (
    #     sigma_max > sigma_max_ir - 1e-2
    # ), f"sigma_max must be greater to its IR value (1%): {sigma_max} vs {sigma_max_ir}"
    assert (
        sigma_max + tol >= sigma_max_ir
    ), f"sigma_max is not greater than its IR value: {sigma_max} vs {sigma_max_ir}"


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("input_channels", [8, 16])
@pytest.mark.parametrize("output_channels", [8, 16])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_standard_configs(kernel_size, input_channels, output_channels, stride, groups):
    """
    test combinations of kernel size, input channels, output channels, stride and groups
    """
    # Test instantiation
    try:
        orthoconv = AdaptiveOrthoConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=(kernel_size // 2, kernel_size // 2),
            padding_mode="circular",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        orthoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
    )


@pytest.mark.parametrize("kernel_size", [2, 3, 4, 5])
@pytest.mark.parametrize("input_channels", [8, 16])
@pytest.mark.parametrize(
    "output_channels", [8, 16]
)  # dilated convolutions are not supported for output_channels < input_channels
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_dilation(kernel_size, input_channels, output_channels, stride, groups):
    """
    test combinations of kernel size, input channels, output channels, stride and groups
    """
    # Test instantiation
    try:
        orthoconv = AdaptiveOrthoConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            dilation=2,
            groups=groups,
            bias=False,
            padding="same",
            padding_mode="circular",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        orthoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
    )


@pytest.mark.parametrize("kernel_size", [2, 3, 4, 5])
@pytest.mark.parametrize("input_channels", [8, 16])
@pytest.mark.parametrize(
    "output_channels", [8, 16]
)  # dilated+strided convolutions are not supported for output_channels < input_channels
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("dilation", [2, 3])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_dilation_strided(
    kernel_size, input_channels, output_channels, stride, dilation, groups
):
    """
    test combinations of kernel size, input channels, output channels, stride and groups
    """
    # Test instantiation
    try:
        orthoconv = AdaptiveOrthoConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=False,
            padding=(
                int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2)),
                int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2)),
            ),
            padding_mode="circular",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        if (output_channels >= input_channels) and (
            ((dilation % stride) == 0) and (stride > 1)
        ):
            # we expect this configuration to raise a ValueError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        if (kernel_size == stride) and (((dilation % stride) == 0) and (stride > 1)):
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        orthoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
    )


@pytest.mark.parametrize("kernel_size", [3, 4, 5])
@pytest.mark.parametrize("input_channels", [2, 4, 32])
@pytest.mark.parametrize("output_channels", [2, 4, 32])
@pytest.mark.parametrize("stride", [2, 4])
@pytest.mark.parametrize("groups", [1])
def test_strided(kernel_size, input_channels, output_channels, stride, groups):
    """
    a more extensive testing when striding is enabled.
    A larger range of cin and cout is used to track errors when cin < cout / stride**2
    ( ie you reduce spatial dimensions but you increase the channel dimensions so
    that you actually increase overall dimension.
    """
    # Test instantiation
    try:
        orthoconv = AdaptiveOrthoConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=((kernel_size - 1) // 2, (kernel_size - 1) // 2),
            padding_mode="circular",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        orthoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
    )


@pytest.mark.parametrize("kernel_size", [2, 4])
@pytest.mark.parametrize("input_channels", [8, 16])
@pytest.mark.parametrize("output_channels", [8, 16])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_even_kernels(kernel_size, input_channels, output_channels, stride, groups):
    """
    test specific to even kernel size
    """
    # Test instantiation
    try:
        orthoconv = AdaptiveOrthoConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding="same",
            padding_mode="circular",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        orthoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
    )


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("input_channels", [4, 8, 32])
@pytest.mark.parametrize("output_channels", [4, 8, 32])
@pytest.mark.parametrize("groups", [1, 2])
def test_rko(kernel_size, input_channels, output_channels, groups):
    """
    test case where stride == kernel size
    """
    # Test instantiation
    try:
        rkoconv = AdaptiveOrthoConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=kernel_size,
            groups=groups,
            bias=False,
            padding=(0, 0),
            padding_mode="zeros",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        rkoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
    )


@pytest.mark.parametrize("kernel_size", [1, 3, 5])
@pytest.mark.parametrize("input_channels", [1, 2])
@pytest.mark.parametrize("output_channels", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1])
def test_depthwise(kernel_size, input_channels, output_channels, stride, groups):
    """
    test combinations of kernel size, input channels, output channels, stride and groups
    """
    # Test instantiation
    try:
        orthoconv = AdaptiveOrthoConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=(kernel_size // 2, kernel_size // 2),
            padding_mode="circular",
            ortho_params=DEFAULT_TEST_ORTHO_PARAMS,
        )
    except Exception as e:
        if input_channels == 1 and output_channels == 1:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        orthoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
    )


def test_invalid_kernel_smaller_than_stride():
    """
    A test to ensure that kernel_size < stride raises an expected ValueError
    """
    with pytest.raises(ValueError, match=r"kernel size must be smaller than stride"):
        AdaptiveOrthoConv2d(
            in_channels=8,
            out_channels=4,
            kernel_size=2,
            stride=3,  # Invalid: kernel_size < stride
            groups=1,
            padding=0,
        )
    with pytest.raises(ValueError, match=r"kernel size must be smaller than stride"):
        BcopRkoConv2d(
            in_channels=8,
            out_channels=4,
            kernel_size=2,
            stride=3,  # Invalid: kernel_size < stride
            groups=1,
            padding=0,
        )
    with pytest.raises(ValueError, match=r"kernel size must be smaller than stride"):
        FastBlockConv2d(
            in_channels=8,
            out_channels=4,
            kernel_size=2,
            stride=3,  # Invalid: kernel_size < stride
            groups=1,
            padding=0,
        )


def test_invalid_dilation_with_stride():
    """
    A test to ensure dilation > 1 while stride > 1 raises an expected ValueError
    """
    with pytest.raises(
        ValueError,
        match=r"dilation must be 1 when stride is not 1",
    ):
        AdaptiveOrthoConv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=2,
            dilation=2,  # Invalid: dilation > 1 while stride > 1
            groups=1,
            padding=0,
        )
    with pytest.raises(
        ValueError,
        match=r"dilation must be 1 when stride is not 1",
    ):
        BcopRkoConv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=2,
            dilation=2,  # Invalid: dilation > 1 while stride > 1
            groups=1,
            padding=0,
        )
    with pytest.raises(
        ValueError,
        match=r"dilation must be 1 when stride is not 1",
    ):
        RKOConv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=2,
            dilation=2,  # Invalid: dilation > 1 while stride > 1
            groups=1,
            padding=0,
        )


@pytest.mark.parametrize("kernel_size", [1, 3])
@pytest.mark.parametrize("input_channels", [8, 16])
@pytest.mark.parametrize("output_channels", [8, 16])
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
        "newtonshultz",
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
        "newtonshultz": NEWTONSHULTZ_ORTHO_PARAMS
    }  # trick to have the actual method name displayed properly if test fails
    # Test instantiation
    try:
        orthoconv = AdaptiveOrthoConv2d(
            kernel_size=kernel_size,
            in_channels=input_channels,
            out_channels=output_channels,
            stride=stride,
            groups=groups,
            bias=False,
            padding=(kernel_size // 2, kernel_size // 2),
            padding_mode="circular",
            ortho_params=ortho_params_dict[ortho_params],
        )
    except Exception as e:
        if kernel_size < stride:
            # we expect this configuration to raise a RuntimeError
            # pytest.skip(f"BCOP instantiation failed with: {e}")
            return
        else:
            pytest.fail(f"BCOP instantiation failed with: {e}")
    check_orthogonal_layer(
        orthoconv,
        groups,
        input_channels,
        kernel_size,
        output_channels,
        (
            output_channels,
            input_channels // groups,
            kernel_size,
            kernel_size,
        ),
        tol=5e-2 if ortho_params.startswith("cholesky") or ortho_params.startswith("newtonshultz") else 1e-3,
        sigma_min_requirement=0.75 if ortho_params.startswith("cholesky") or ortho_params.startswith("newtonshultz") else 0.95,
    )
