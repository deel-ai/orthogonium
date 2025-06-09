import pytest
import os
import numpy as np
import torch
from torch import from_numpy
from torch.nn import Sequential as tSequential

from orthogonium.layers import LayerCentering2D, BatchCentering2D
from orthogonium.layers import BatchCentering, BatchLipNorm
from orthogonium.layers import SharedLipFactory


@pytest.mark.parametrize(
    "layer_fn, num_features, orthogonal",
    [
        (lambda: LayerCentering2D(num_features=4), 4, False),
        # (lambda: BatchCentering2D(num_features=4), 4, False),
    ],
)
@pytest.mark.parametrize(
    "mean, std",
    [
        (0, 1),  # Standard Normal Distribution
        (5, 2),  # Higher mean and variance
        (-3, 1),  # Shifted mean
        (0, 0.1),  # Low variance
        (10, 5),  # Very high variance
    ],
)
def test_lipschitz_constant_with_various_distributions(
    layer_fn, num_features, orthogonal, mean, std
):
    """
    Test if the layer satisfies the Lipschitz property when the input
    comes from distributions with different means and variances.
    """
    layer = layer_fn()
    layer.train()  # Set layer to training mode

    batch_size, h, w = 8, 8, 8  # Input dimensions

    # Generate input tensor from a specific distribution
    x = torch.randn(batch_size, num_features, h, w) * std + mean
    x.requires_grad_(True)  # Enable gradient tracking

    y = layer(x)
    x.requires_grad_(True)  # Enable gradient tracking

    # Compute the Jacobian using jacrev
    batch_jacobian = torch.func.jacrev(layer)(x)

    # Reshape the Jacobian to match the desired shape
    batch_size = x.shape[0]
    ydim = torch.prod(torch.tensor(y.shape)).item()
    xdim = torch.prod(torch.tensor(x.shape)).item()

    jacobian = batch_jacobian.view(ydim, xdim)

    # Validate Lipschitz constant
    if orthogonal:
        singular_values = torch.linalg.svdvals(jacobian)
        assert singular_values.max() <= 1 + 1e-4, (
            f"Lipschitz constraint violated for input distribution with mean={mean}, std={std}; "
            f"max singular value: {singular_values.max()}"
        )
        assert (
            singular_values.min() >= 1 - 1e-4
        ), f"Orthogonality constraint violated for input distribution with mean={mean}, std={std}; "
    else:
        lipschitz_constant = torch.linalg.matrix_norm(jacobian, ord=2).item()
        assert lipschitz_constant <= 1 + 1e-4, (
            f"Lipschitz constraint violated for input distribution with mean={mean}, std={std}; "
            f"Lipschitz constant: {lipschitz_constant}"
        )


def generate_k_lip_model(layer_type: type, layer_params: dict, input_shape=None, k=1):
    """
    build a model with a single layer of given type, with defined lipshitz factor.

    Args:
        layer_type: the type of layer to use
        layer_params: parameter passed to constructor of layer_type
        input_shape: the shape of the input
        k: lipshitz factor of the function

    Returns:
        a torch Model with a single layer.

    """
    if issubclass(layer_type, tSequential):
        layers_list = [lay for lay in layer_params["layers"] if lay is not None]
        assert len(layers_list) > 0
        if k is not None:
            model = layer_type(*layers_list, k_coef_lip=k)
        else:
            model = layer_type(*layers_list)
        # model.set_klip_factor(k)
        return model
    """if issubclass(layer_type, LipschitzLayer):
        layer_params["k_coef_lip"] = k
    """
    layer = layer_type(
        **layer_params
    )  # get_instance_framework(layer_type, layer_params)
    assert isinstance(layer, torch.nn.Module)  # or isinstance(layer, Linear)
    return tSequential(layer)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def check_serialization(layer_type, layer_params, input_shape=(10,), norm=False):
    if norm:
        factory = SharedLipFactory()
        layer_params["factory"] = factory
    m = generate_k_lip_model(layer_type, layer_params, input_shape=input_shape, k=1)
    if m is None:
        pytest.skip()
    loss = torch.nn.CrossEntropyLoss()

    if len(list(m.parameters())) != 0:
        optimizer = torch.optim.SGD(**{"params": m.parameters()})

    name = layer_type.__class__.__name__
    path = os.path.join("logs", "normalization", name)
    xnp = np.random.uniform(-10, 10, (255,) + input_shape)
    x = torch.tensor(xnp, dtype=torch.float32)
    y1 = m(x)

    parent_dirpath = os.path.split(path)[0]
    if not os.path.exists(parent_dirpath):
        os.makedirs(parent_dirpath)
    torch.save(m.state_dict(), path)
    if norm:
        factory = SharedLipFactory()
        layer_params["factory"] = factory

    m2 = generate_k_lip_model(layer_type, layer_params, input_shape, k=1)
    m2.load_state_dict(torch.load(path))
    y2 = m2(x)
    np.testing.assert_allclose(to_numpy(y1), to_numpy(y2))


def train(
    train_dl,
    model,
    loss_fn,
    optimizer,
    epoch,
    batch_size,
    steps_per_epoch,
    callbacks=[],
):
    for epoch in range(epoch):
        model.train()
        for _ in range(steps_per_epoch):
            # for xb, yb in train_dl:
            xb, yb = next(train_dl)
            xb, yb = from_numpy(xb), from_numpy(yb)
            pred = model(xb.float())
            loss = loss_fn(pred, yb.float())
            # compute gradient and do SGD step
            if optimizer is not None:  # in case of no parameter in the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


"""@pytest.mark.skipif(
    hasattr(BatchLipNorm, "unavailable_class"),
    reason="BatchLipNorm not available",
)"""


@pytest.mark.parametrize(
    "size, input_shape, bias, norm",
    [
        (4, (3, 4), False, False),
        (4, (3, 4), False, True),
        (4, (3, 4), True, False),
        (4, (3, 4), True, True),
        (4, (3, 4, 8, 8), False, False),
        (4, (3, 4, 8, 8), False, True),
        (4, (3, 4, 8, 8), True, False),
        (4, (3, 4, 8, 8), True, True),
    ],
)
def test_BatchLipNorm(size, input_shape, bias, norm):
    """evaluate layerbatch centering"""
    # input_shape = uft.to_framework_channel(input_shape)
    xnp = np.arange(np.prod(input_shape)).reshape(input_shape)
    factory = None
    if norm:
        factory = SharedLipFactory()
    bn = BatchLipNorm(**{"num_features": size, "bias": bias, "factory": factory})
    # bn_mom = bn.momentum
    if len(input_shape) == 2:
        mean_x = np.mean(xnp, axis=0)
        var_x = np.var(xnp, axis=0, ddof=1)
        mean_shape = (1, size)
    else:
        mean_x = np.mean(xnp, axis=(0, 2, 3))
        var_x = np.var(xnp, axis=(0, 2, 3), ddof=1)
        mean_shape = (1, size, 1, 1)
    x = torch.tensor(xnp, dtype=torch.float32)
    y = bn(x)
    scale_factor = current_scale_factor = 1.0
    if norm:
        print("gt var ", var_x)
        scale_factor = np.max(np.sqrt(var_x))
    np.testing.assert_allclose(bn.get_running_mean(), mean_x, atol=1e-5)
    np.testing.assert_allclose(bn.get_scaling_factor(), scale_factor, atol=1e-5)
    np.testing.assert_allclose(
        to_numpy(y), (xnp - np.reshape(mean_x, mean_shape)) / scale_factor, atol=1e-5
    )
    y = bn(2 * x)
    new_runningmean = (
        mean_x + 2 * mean_x
    ) / 2.0  # mean_x * (1 - bn_mom) + 2 * mean_x * bn_mom
    # new_runningvar = (var_x + 4 * var_x)/2.0 #mean_x * (1 - bn_mom) + 2 * mean_x * bn_mom
    conc_xnp = np.concatenate([xnp, 2 * xnp], axis=0)
    if len(input_shape) == 2:
        new_runningvar = np.var(conc_xnp, axis=0, ddof=1)
    else:
        new_runningvar = np.var(
            conc_xnp, axis=(0, 2, 3), ddof=1
        )  # (var_x + 4 * var_x + 0.5*(mean_x-2 * mean_x)**2)/2.0 #mean_x * (1 - bn_mom) + 2 * mean_x * bn_mom
    if norm:
        print("gtnew_runningvar ", new_runningvar)
        current_scale_factor = np.max(np.sqrt(4 * var_x))
        scale_factor = np.max(np.sqrt(new_runningvar))
    np.testing.assert_allclose(
        bn.get_running_mean(), new_runningmean, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        bn.get_scaling_factor(), scale_factor, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(y),
        (2 * xnp - 2 * np.reshape(mean_x, mean_shape)) / current_scale_factor,
        atol=1e-5,
        rtol=1e-5,
    )  # keep substract batch mean
    print("start in eval mode")
    bn.eval()
    y = bn(2 * x)
    np.testing.assert_allclose(
        bn.get_running_mean(), new_runningmean, atol=1e-5
    )  # eval mode running mean freezed
    np.testing.assert_allclose(
        bn.get_scaling_factor(), scale_factor, atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        to_numpy(y),
        (2 * xnp - np.reshape(new_runningmean, mean_shape)) / scale_factor,
        atol=1e-5,
        rtol=1e-5,
    )  # eval mode use running_mean


@pytest.mark.parametrize(
    "norm_type",
    [
        BatchLipNorm,
    ],
)
@pytest.mark.parametrize(
    "norm",
    [False, True],
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (10, (10,), False),
        (10, (10,), True),
        (7, (7, 8, 8), False),
        (7, (7, 8, 8), True),
    ],
)
def test_Normalization_serialization(norm_type, size, input_shape, bias, norm):
    # Check serialization
    if hasattr(norm_type, "unavailable_class"):
        pytest.skip(f"{norm_type} not available")
    check_serialization(
        norm_type,
        layer_params={"num_features": size, "bias": bias},
        input_shape=input_shape,
        norm=norm,
    )


def linear_generator(batch_size, input_shape: tuple):
    """
    Generate data according to a linear kernel
    Args:
        batch_size: size of each batch
        input_shape: shape of the desired input

    Returns:
        a generator for the data

    """
    input_shape = tuple(input_shape)
    while True:
        # pick random sample in [0, 1] with the input shape
        batch_x = np.array(
            np.random.uniform(-10, 10, (batch_size,) + input_shape), dtype=np.float16
        )
        # apply the k lip linear transformation
        batch_y = batch_x
        yield batch_x, batch_y


@pytest.mark.parametrize(
    "norm_type",
    [
        BatchLipNorm,
    ],
)
@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (10, (10,), True),
        (7, (7, 8, 8), True),
    ],
)
@pytest.mark.parametrize(
    "norm",
    [False, True],
)
def test_Normalization_bias(norm_type, size, input_shape, bias, norm):
    if hasattr(norm_type, "unavailable_class"):
        pytest.skip(f"{norm_type} not available")
    factory = None
    if norm:
        factory = SharedLipFactory()
    m = generate_k_lip_model(
        norm_type,
        layer_params={"num_features": size, "bias": bias, "factory": factory},
        input_shape=input_shape,
        k=1,
    )
    if m is None:
        pytest.skip()
    optimizer = torch.optim.SGD(**{"params": m.parameters()})
    loss = torch.nn.CrossEntropyLoss()

    batch_size = 10
    bb = to_numpy(m[0].bias)
    sf = to_numpy(m[0].get_scaling_factor())
    np.testing.assert_allclose(bb, np.zeros((size,)), atol=1e-5)
    np.testing.assert_allclose(sf, 1.0, atol=1e-5)

    traind_ds = linear_generator(batch_size, input_shape)
    train(
        traind_ds,
        m,
        loss,
        optimizer,
        2,
        batch_size,
        steps_per_epoch=10,
    )

    bb = to_numpy(m[0].bias)
    sf = to_numpy(m[0].get_scaling_factor())
    assert np.linalg.norm(bb) != 0.0
    if norm:
        assert np.linalg.norm(sf) != 1.0
    else:
        assert np.linalg.norm(sf) == 1.0


"""@pytest.mark.skipif(
    hasattr(BatchLipNorm, "unavailable_class"),
    reason="BatchLipNorm not available",
)"""


@pytest.mark.parametrize(
    "size, input_shape, bias",
    [
        (4, (3, 4), False),
        (4, (3, 4), True),
        (4, (3, 4, 8, 8), False),
        (4, (3, 4, 8, 8), True),
        (13, (64, 13, 8, 8), False),
        (13, (64, 13, 8, 8), True),
    ],
)
@pytest.mark.parametrize(
    "norm",
    [False, True],
)
@pytest.mark.parametrize(
    "type_seq",
    [0, 1, 2],
)
def test_BatchLipNorm_runningmean(size, input_shape, bias, norm, type_seq):
    """evaluate batch centering convergence of running mean"""
    # input_shape = uft.to_framework_channel(input_shape)
    # start with 0 to set up running mean to zero
    if type_seq == 0:
        xnp = np.zeros(input_shape)
        gt_mean = 0.0
        gt_var = 1.0
        epochs = 2
    elif type_seq >= 1:
        epochs = 20
        xnp = np.random.normal(0.0, 1.0, input_shape)
        if len(input_shape) == 2:
            mean_x = np.mean(xnp, axis=0)
            var_x = np.var(xnp, axis=0, ddof=1)
            num_elem = input_shape[0]
        else:
            mean_x = np.mean(xnp, axis=(0, 2, 3))
            var_x = np.var(xnp, axis=(0, 2, 3), ddof=1)
            num_elem = input_shape[0] * input_shape[2] * input_shape[3]
        if type_seq == 1:
            gt_mean = mean_x
            gt_var = var_x * (num_elem - 1) * epochs / (epochs * num_elem - 1)
        else:
            conc_xnp = np.concatenate([xnp, 3 * xnp], axis=0)
            if len(input_shape) == 2:
                new_runningvar = np.var(conc_xnp, axis=0, ddof=1)
            else:
                new_runningvar = np.var(conc_xnp, axis=(0, 2, 3), ddof=1)
            gt_mean = (mean_x + 3 * mean_x) / 2.0
            print("gt mean_x ", mean_x, "new_runningvar ", new_runningvar)
            gt_var = (
                new_runningvar
                * (2 * num_elem - 1)
                * epochs
                / (2 * epochs * num_elem - 1)
            )

    factory = None
    if norm:
        factory = SharedLipFactory()
    bn = BatchLipNorm(**{"num_features": size, "bias": bias, "factory": factory})
    x = torch.tensor(xnp, dtype=torch.float32)

    for _ in range(epochs):
        y = bn(x)  # noqa: F841
        if type_seq == 2:
            y = bn(3 * x)
    # noqa: F841

    np.testing.assert_allclose(bn.get_running_mean(), gt_mean, atol=1e-5)
    # constant value => no scale factor
    if (type_seq == 0) or norm:
        print(
            "get_scaling_factor ",
            bn.get_scaling_factor(False),
            " var_x ",
            gt_var,
            np.sqrt(np.max(gt_var)),
        )
        np.testing.assert_allclose(
            bn.get_scaling_factor(), np.sqrt(np.max(gt_var)), atol=1e-5
        )
