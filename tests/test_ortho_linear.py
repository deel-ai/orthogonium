import pytest
import torch
import numpy as np

from orthogonium.layers import UnitNormLinear
from orthogonium.layers.linear import OrthoLinear
from orthogonium.reparametrizers import (
    DEFAULT_TEST_ORTHO_PARAMS,
    CHOLESKY_ORTHO_PARAMS,
    CHOLESKY_STABLE_ORTHO_PARAMS,
    QR_ORTHO_PARAMS,
    NEWTONSHULTZ_ORTHO_PARAMS,
)

device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _compute_sv_weight_matrix(layer):
    """
    Computes the singular values of the weight matrix of a linear layer.
    """
    with torch.no_grad():
        try:
            svs = torch.linalg.svdvals(layer.weight)
            return svs.min().item(), svs.max().item(), (svs.mean() / svs.max()).item()
        except np.linalg.LinAlgError:
            pytest.fail("SVD computation failed.")


@pytest.mark.parametrize("input_features", [16, 32, 64])
@pytest.mark.parametrize("output_features", [16, 32, 64])
@pytest.mark.parametrize("bias", [True, False])
def test_ortho_linear_instantiation(input_features, output_features, bias):
    """
    Test that OrthoLinear can be instantiated and has the correct weight properties.
    """
    try:
        layer = OrthoLinear(input_features, output_features, bias=bias).to(device)
        assert layer.weight.shape == (output_features, input_features), (
            f"Weight shape mismatch: {layer.weight.shape} "
            f"!= {(output_features, input_features)}"
        )
    except Exception as e:
        pytest.fail(f"OrthoLinear instantiation failed: {e}")


@pytest.mark.parametrize(
    "input_features, output_features", [(16, 16), (16, 32), (32, 32)]
)
@pytest.mark.parametrize("bias", [True, False])
def test_ortho_linear_singular_values(input_features, output_features, bias):
    """
    Test the singular values of the weight matrix in OrthoLinear.
    """
    layer = OrthoLinear(input_features, output_features, bias=bias).to(device)
    sigma_min, sigma_max, stable_rank = layer.singular_values()
    assert sigma_max <= 1.01, "Maximum singular value exceeds 1.01"
    assert 0.95 <= sigma_min <= 1.05, "Minimum singular value not close to 1"
    assert 0.98 <= stable_rank <= 1.02, "Stable rank not close to 1"


@pytest.mark.parametrize(
    "input_features, output_features", [(16, 16), (16, 32), (32, 32)]
)
def test_ortho_linear_norm_preservation(input_features, output_features):
    """
    Test that the OrthoLinear layer preserves norm as expected.
    """
    layer = OrthoLinear(input_features, output_features, bias=False).to(device)
    inp = torch.randn(8, input_features).to(device)  # Batch of 8
    with torch.no_grad():
        output = layer(inp)
        inp_norm = torch.norm(inp, dim=-1).mean().item()
        out_norm = torch.norm(output, dim=-1).mean().item()
        assert (
            abs(inp_norm - out_norm) < 1e-2
        ), f"Norm preservation failed: input norm {inp_norm}, output norm {out_norm}"


@pytest.mark.parametrize("input_features", [16, 32])
@pytest.mark.parametrize("output_features", [16, 32])
def test_ortho_linear_training(input_features, output_features):
    """
    Test backpropagation and training of OrthoLinear.
    """
    layer = OrthoLinear(input_features, output_features, bias=True).to(device)
    layer.train()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)
    for _ in range(10):
        optimizer.zero_grad()
        inp = torch.randn(8, input_features).to(device)
        output = layer(inp)
        loss = -output.mean()
        loss.backward()
        optimizer.step()

    try:
        layer.eval()
        with torch.no_grad():
            svs_after_training = torch.linalg.svdvals(layer.weight).cpu().numpy()
            assert (
                svs_after_training.max() <= 1.05
            ), f"Max singular value after training too high: {svs_after_training.max()}"
    except Exception as e:
        pytest.fail(f"Training or SVD computation failed: {e}")


@pytest.mark.parametrize("input_features", [16, 32])
@pytest.mark.parametrize("output_features", [16, 32])
def test_ortho_linear_impulse_response(input_features, output_features):
    """
    Compare singular values from impulse response and weight matrix SVD.
    """
    layer = OrthoLinear(input_features, output_features, bias=False).to(device)
    sigma_min_wr, sigma_max_wr, stable_rank_wr = layer.singular_values()
    sigma_min_ir, sigma_max_ir, stable_rank_ir = _compute_sv_weight_matrix(layer)

    tol = 1e-4
    assert abs(sigma_min_wr - sigma_min_ir) < tol, (
        f"Impulse response min singular value mismatch: "
        f"{sigma_min_wr} vs {sigma_min_ir}"
    )
    assert abs(sigma_max_wr - sigma_max_ir) < tol, (
        f"Impulse response max singular value mismatch: "
        f"{sigma_max_wr} vs {sigma_max_ir}"
    )
    assert (
        abs(stable_rank_wr - stable_rank_ir) < tol
    ), f"Impulse response stable rank mismatch: {stable_rank_wr} vs {stable_rank_ir}"


@pytest.mark.parametrize("input_features", [16, 32])  # Adjust input sizes as needed
@pytest.mark.parametrize("output_features", [16, 32])
@pytest.mark.parametrize(
    "orthparams_name, orthparams",
    [
        ("default", DEFAULT_TEST_ORTHO_PARAMS),
        ("cholesky", CHOLESKY_ORTHO_PARAMS),
        ("cholesky_stable", CHOLESKY_STABLE_ORTHO_PARAMS),
        ("qr", QR_ORTHO_PARAMS),
        ("newtonshultz", NEWTONSHULTZ_ORTHO_PARAMS)
    ],
)
def test_ortho_linear_with_orthparams(
    input_features, output_features, orthparams_name, orthparams
):
    """
    Test OrthoLinear under different orthparams settings.
    """
    try:
        layer = OrthoLinear(
            input_features, output_features, bias=True, ortho_params=orthparams
        ).to(device)

        # Validate weight shape
        assert layer.weight.shape == (output_features, input_features), (
            f"Weight shape mismatch for {orthparams_name}: "
            f"{layer.weight.shape} != {(output_features, input_features)}"
        )

        # Validate singular values
        sigma_min, sigma_max, stable_rank = layer.singular_values()
        print(f"sigma_min: {sigma_min}, sigma_max: {sigma_max}, stable_rank: {stable_rank}")
        # Add precision tolerances for different orthparams
        tol = 1e-2 if orthparams_name.startswith("cholesky") or orthparams_name.startswith("newtonshultz") else 1e-3
        assert (
            sigma_max <= 1 + tol
        ), f"Max singular value exceeds tolerance for {orthparams_name}"
        assert (
            1 - tol <= sigma_min <= 1 + tol
        ), f"Min singular value out of tolerance for {orthparams_name}"
        assert (
            0.98 <= stable_rank <= 1.02
        ), f"Stable rank out of bounds for {orthparams_name}"

    except Exception as e:
        pytest.fail(f"Test failed for orthparams '{orthparams_name}': {e}")


@pytest.mark.parametrize("input_features", [16, 32])
@pytest.mark.parametrize("output_features", [16, 32])
@pytest.mark.parametrize("bias", [True, False])
def test_unitnorm_linear_instantiation(input_features, output_features, bias):
    """
    Test that UnitNormLinear can be instantiated and has the correct weight properties.
    """
    try:
        layer = UnitNormLinear(input_features, output_features, bias=bias).to(device)
        assert layer.weight.shape == (
            output_features,
            input_features,
        ), f"Weight shape mismatch: {layer.weight.shape} != {(output_features, input_features)}"
    except Exception as e:
        pytest.fail(f"UnitNormLinear instantiation failed: {e}")


@pytest.mark.parametrize("input_features", [16, 32])
@pytest.mark.parametrize("output_features", [16, 32])
def test_unitnorm_linear_weight_normalization(input_features, output_features):
    """
    Test that the weight rows of UnitNormLinear are unit-normalized.
    """
    layer = UnitNormLinear(input_features, output_features).to(device)
    with torch.no_grad():
        frobenius_norms = torch.linalg.norm(layer.weight, dim=1)
        assert torch.allclose(
            frobenius_norms, torch.ones_like(frobenius_norms), atol=1e-4
        ), f"Row norms are not equal to 1: {frobenius_norms}"


@pytest.mark.parametrize("input_features", [16, 32])
@pytest.mark.parametrize("output_features", [16, 32])
@pytest.mark.parametrize("batch_size", [8, 16])
def test_unitnorm_linear_lipschitz_property(
    input_features, output_features, batch_size
):
    """
    Test if each output of UnitNormLinear satisfies the 1-Lipschitz property.
    """
    layer = UnitNormLinear(input_features, output_features).to(device)
    layer.eval()

    x = torch.randn(batch_size, input_features, requires_grad=True).to(device)
    y = layer(x)

    # Calculate Jacobian
    jacobian = []
    for i in range(y.numel()):  # Loop over each output feature
        grad_output = torch.zeros_like(y)
        grad_output.view(-1)[i] = 1  # Assign 1 to the i-th output for derivative calc
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=grad_output,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        jacobian.append(gradients.view(-1).detach().cpu().numpy())

    jacobian = torch.tensor(jacobian).view(y.numel(), x.numel())  # Construct Jacobian

    # Validate Lipschitz norm per row
    row_norms = torch.linalg.norm(jacobian, dim=1)
    assert torch.all(
        row_norms <= 1.0 + 1e-4
    ), f"Some rows do not satisfy 1-Lipschitz: {row_norms}"
    assert torch.all(
        row_norms >= 1.0 - 1e-4
    ), f"Some rows are norm preseving: {row_norms}"


@pytest.mark.parametrize("input_features", [16, 32])
@pytest.mark.parametrize("output_features", [16, 32])
def test_unitnorm_linear_training(input_features, output_features):
    """
    Test backpropagation and training of UnitNormLinear.
    """
    layer = UnitNormLinear(input_features, output_features, bias=True).to(device)
    layer.train()
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.01)

    for _ in range(10):
        optimizer.zero_grad()
        inp = torch.randn(8, input_features).to(device)  # Batch size: 8
        output = layer(inp)
        loss = -output.mean()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        row_norms = torch.linalg.norm(layer.weight, dim=1)
        # Ensure row norms are still ~1 after training
        assert torch.allclose(
            row_norms, torch.ones_like(row_norms), atol=1e-4
        ), f"Row norms after training not equal to 1: {row_norms}"
