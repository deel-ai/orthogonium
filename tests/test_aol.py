import pytest
import torch
from torch import nn
from orthogonium.layers.conv.AOL.aol import AOLConv2D, AOLConvTranspose2D
from orthogonium.layers.conv.singular_values import get_conv_sv
import numpy as np

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


@pytest.mark.parametrize("convclass", [AOLConv2D, AOLConvTranspose2D])
@pytest.mark.parametrize("in_channels", [4, 8])
@pytest.mark.parametrize("out_channels", [4, 8])
@pytest.mark.parametrize("kernel_size", [2, 3])
@pytest.mark.parametrize("groups", [1, 4])
@pytest.mark.parametrize("niter", [1, 3])
def test_lipschitz_layers(
    convclass, in_channels, out_channels, kernel_size, groups, niter
):
    """
    Generalized test for layers in the AOL module to check Lipschitz constraints.
    """
    tol = 2e-2
    # Initialize layer
    layer = convclass(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        bias=False,
        niter=niter,
    )
    # re-init weight as it is orthogonal by default
    layer.weight = torch.nn.init.uniform_(layer.weight)

    # Define input and target tensors
    x = torch.randn((4, in_channels, 8, 8), requires_grad=True)  # Input

    # Pre-optimization Lipschitz constant (if applicable)
    # pre_lipschitz_constant = get_conv_sv(layer, n_iter=3, agg_groups=True)
    sv_min_ir_pre, sv_max_ir_pre, sr_ir_pre = _compute_sv_impulse_response_layer(
        layer, (in_channels, 8, 8)
    )
    print("Pre-optimization Lipschitz constant:")
    print(
        f"Impulse response: sv min={sv_min_ir_pre:.3f}, sv_max={sv_max_ir_pre:.3f}, stable rank={sr_ir_pre:.3f}"
    )
    # print(
    #     f"delattre 24 meth: sv min={0.000:.3f}, sv_max={pre_lipschitz_constant[0]:.3f}, stable rank={pre_lipschitz_constant[1]:.3f}"
    # )
    # Define optimizer and loss function
    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-3)

    # Perform a few optimization steps
    for _ in range(10):  # Run 10 optimization steps
        optimizer.zero_grad()
        output = layer(x)
        loss = -torch.sum(torch.square(output))
        loss.backward()
        optimizer.step()

    # # Post-optimization Lipschitz constant (if applicable)
    # post_lipschitz_constant = get_conv_sv(layer, n_iter=3, agg_groups=True)

    sv_min_ir_post, sv_max_ir_post, sr_ir_post = _compute_sv_impulse_response_layer(
        layer, (in_channels, 8, 8)
    )
    print("Post-optimization Lipschitz constant:")
    print(
        f"Impulse response: sv min={sv_min_ir_post:.3f}, sv_max={sv_max_ir_post:.3f}, stable rank={sr_ir_post:.3f}"
    )
    # print(
    #     f"delattre 24 meth: sv min={0.000:.3f}, sv_max={post_lipschitz_constant[0]:.3f}, stable rank={post_lipschitz_constant[1]:.3f}"
    # )
    assert sv_max_ir_pre <= 1 + tol, "Pre-optimization Lipschitz constant violation."
    assert sv_max_ir_post <= 1 + tol, "Post-optimization Lipschitz constant violation."
