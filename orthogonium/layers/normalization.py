import torch
import torch.nn as nn
import abc
import copy
import torch.distributed as dist
from torch.nn import Sequential as TorchSequential
from typing import Optional
from collections import OrderedDict
import os


class LayerCentering2D(nn.Module):
    def __init__(self, num_features):
        super(LayerCentering2D, self).__init__()
        self.bias = nn.Parameter(
            torch.zeros((1, num_features, 1, 1)), requires_grad=True
        )

    def forward(self, x):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        return x - mean + self.bias


class BatchCentering(nn.Module):
    def __init__(
        self,
        num_features: int = 1,
        dim: Optional[tuple] = None,
        bias: bool = True,
    ):
        super(BatchCentering, self).__init__()
        self.dim = dim
        self.num_features = num_features

        self.register_buffer("running_mean", torch.zeros((num_features,)))
        self.register_buffer("running_num_batches", torch.zeros((1,)))
        if bias:
            self.bias = nn.Parameter(torch.zeros((num_features,)), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.first = True

    def reset_states(self):
        self.running_mean.zero_()
        self.running_num_batches.zero_()

    # compute average of running values
    def update_running_values(self):
        if self.running_num_batches > 1:
            self.running_mean = self.running_mean / self.running_num_batches
            self.running_num_batches = self.running_num_batches.zero_() + 1.0

    def get_running_mean(self, training=False, update=True):
        """Retrieve the running mean in eval mode."""
        """ If update is True, update the running values"""

        assert training == False, "Only in eval mode"
        # case asking for running mean before a step
        if self.running_num_batches == 0:
            return torch.zeros(self.running_mean.shape).to(self.running_mean.device)
        if update and (self.running_num_batches > 1):
            self.update_running_values()
        return self.running_mean / self.running_num_batches

    def forward(self, x):
        if self.dim is None:  # (0,2,3) for 4D tensor; (0,) for 2D tensor
            self.dim = (0,) + tuple(range(2, len(x.shape)))
        mean_shape = (1, -1) + (1,) * (len(x.shape) - 2)
        if self.training:
            if self.first:
                self.reset_states()
                self.first = False
            # compute local mean (on batch of a single GPU)
            mean = x.mean(dim=self.dim)
            agrregated_mean = mean.clone().detach()
            # on a single GPU this value is always 1
            num_batches = self.running_num_batches.clone().detach().zero_() + 1.0
            # for multiGPU aggregate mean and num_batches
            if dist.is_initialized():
                dist.all_reduce(agrregated_mean.detach(), op=dist.ReduceOp.SUM)
                dist.all_reduce(num_batches.detach(), op=dist.ReduceOp.SUM)
            with torch.no_grad():
                self.running_mean += agrregated_mean
                self.running_num_batches += num_batches

        else:
            mean = self.get_running_mean(self.training)
        if self.bias is not None:
            return x - mean.view(mean_shape) + self.bias.view(mean_shape)
        else:
            return x - mean.view(mean_shape)


BatchCentering2D = BatchCentering


class SharedLipFactory:
    def __init__(self, eps: float = 1e-5):
        self.buffers_name2module = {}
        self.eps = eps

    def get_shared_buffer(self, module, buffer_name, shape=(1,)):
        """Retrieve or create a shared buffer within the model."""
        if buffer_name not in self.buffers_name2module:
            self.buffers_name2module[buffer_name] = []
        buffer = torch.ones(shape)
        module.register_buffer("running_" + buffer_name, buffer)
        current_buffer = torch.ones(shape)
        module.register_buffer("current_" + buffer_name, current_buffer)
        self.buffers_name2module[buffer_name].append(module)

    def get_var_value(self, module, training, update=True):
        """Retrieve the current variance either local or global."""
        """ current_mean and current_sample_num are required in training mode (for local variance)"""
        """update is only for unit testing (getting values without averaging)"""
        assert len(self.buffers_name2module) == 1, "Only one buffer type is supported"
        buffer_name = list(self.buffers_name2module.keys())[0]
        if training:
            # rectified local variance using local buffers:
            # var_sum is sum_on_batch(x_i^2)/current_sample_num with current_sample_num batch size
            # mean_sum is sum_on_batch(x_i)/current_sample_num
            # variance = (var_sum - (mean_sum*mean_sum))*current_sample_num/(current_sample_num-1)
            var_sum = getattr(module, "current_" + buffer_name)
            mean_sum = getattr(module, "current_mean")
            current_sample_num = getattr(module, "local_num_elements")
            assert (
                mean_sum is not None
            ), "current_mean should be provided in training mode"
            assert (
                current_sample_num is not None
            ), "current_sample_num should be provided in training mode"
            var_factor = current_sample_num / (current_sample_num - 1)
            num_batches = 1.0
        else:
            # rectified variance using running buffers:
            # var_sum is sum_on_epoch(x_i^2)/total_num_samples with total_num_samples number of samples
            # mean_sum is sum_on_epoch(x_i)/total_num_samples
            # variance = (var_sum - (mean_sum*mean_sum))*total_num_samples/(total_num_samples-1)
            mean_sum = module.get_running_mean(update=update)
            num_batches = getattr(module, "running_num_batches")
            # Need to divide by num_batches to get the average on epoch with multigpu
            var_sum = getattr(module, "running_" + buffer_name) / num_batches
            total_num_samples = getattr(module, "total_num_samples")
            var_factor = total_num_samples / (total_num_samples - 1)

        if num_batches == 0:
            return torch.ones((1,))
        var = (var_sum - (mean_sum * mean_sum)) * var_factor
        var = torch.where(var < self.eps, torch.ones(var.shape).to(var.device), var)
        return var.max()

    def get_current_product_value(self, training):
        """Retrieve the current product of the shared buffers."""
        assert len(self.buffers_name2module) == 1, "Only one buffer type is supported"
        buffer_name = list(self.buffers_name2module.keys())[0]
        buffers = [
            module.get_scaling_factor(training)
            for module in self.buffers_name2module[buffer_name]
        ]
        return torch.prod(torch.stack(buffers))


class ScaledLipschitzModule(abc.ABC):
    """
    This class allow to set learnable lipschitz parameter of a layer.

    """

    def __init__(
        self, factory: Optional[SharedLipFactory] = None, factor_name: str = "var"
    ):

        # Factory of factors
        self.factory = factory
        # factor name:
        self.factor_name = factor_name

    """Retrieve the scaling_factor of the layer (max(sqrt(variance))."""
    """ current_mean and current_sample_num are required in training mode (for local variance)"""
    """ update is only for unit testing (getting values without averaging)"""

    def get_scaling_factor(self, training: bool = False, update=True):
        var = self.get_variance_factor(training, update=update)
        return var.sqrt()

    def get_variance_factor(self, training: bool, update=True):
        if self.factory is None:
            return torch.ones((1,))
        else:
            return self.factory.get_var_value(self, training, update=update)

    @abc.abstractmethod
    def vanilla_export(self, lambda_cumul):
        """
        Convert this layer to a corresponding vanilla torch layer (when possible).
        Based on the cumulated scaling factor of the previous layers.
        Returns:
             A vanilla torch version of this layer.
        """
        pass


class ScaleBiasLayer(nn.Module):
    def __init__(
        self,
        scalar=1.0,
        num_features: int = 1,
        bias: bool = True,
    ):
        """
        A PyTorch layer that multiplies the input by a fixed scalar.
        and add a bias
        :param scalar: The scalar multiplier (non-learnable).
        :param size: number of features in the input tensor
        :param bias: if `True`, adds a learnable bias to the output
        of shape (size,). Default: `True`
        """
        super(ScaleBiasLayer, self).__init__()
        self.scalar = scalar
        self.num_features = num_features
        if bias:
            self.bias = nn.Parameter(torch.zeros((num_features,)), requires_grad=True)

    def forward(self, x):
        if self.bias is not None:
            return x * self.scalar + self.bias
        else:
            return x * self.scalar


class BatchLipNorm(nn.Module, ScaledLipschitzModule):
    r"""
    Applies Batch Normalization with a single learnable parameter for normalization  over a 2D, 3D, 4D input.

    .. math::

        y_i = \frac{x_i - \mathrm{E}[x_i]}{\lambda} + \beta_i
        \lambda = max_i(\sqrt{\mathrm{Var}[x_i] + \epsilon})

    The mean is calculated per-dimension over the mini-batches and
    other dimensions excepted the feature/channel dimension.
    Contrary to BatchNorm, the normalization factor :math:`\lambda`
    is common to all the features of the input tensor.
    This learnable parameter is given by a SharedLipFactory instance.
    This layer uses statistics computed from input data in
    training mode and  a constant in evaluation mode computed as
    the running mean on training samples.
    :math:`\beta` is a learnable parameter vectors
    of size `C` (where `C` is the number of features or channels of the input).
    that can be applied after the mean subtraction.
    This layer is :math:`\frac{1}{\lambda}`-Lipschitz and should be used
    only in a sequential model with a last layer that compensate the product
    of the normalization factors.

    Args:
        size: number of features in the input tensor
        dim: dimensions over which to compute the mean
        (default ``input.mean((0, -2, -1))`` for a 4D tensor).
        momentum: the value used for the running mean computation
        centering: if `True`, subtracts the mean from the input
        bias: if `True`, adds a learnable bias to the output
        of shape (size,). Default: `True`
        factory: a SharedLipFactory instance that provides the scaling factor
        if `None`, the scaling factor is set to 1.0. The SharedLipFactory enable to
        compensate the product of the normalization factors in a sequential model.

    Shape:
        - Input: :math:`(N, size, *)`
        - Output: :math:`(N, size, *)` (same shape as input)

    """

    def __init__(
        self,
        num_features: int = 1,
        dim: Optional[tuple] = None,
        momentum: float = 0.05,
        centering: bool = True,
        bias: bool = True,
        factory: Optional[SharedLipFactory] = None,
        eps: float = 1e-5,
    ):
        nn.Module.__init__(self)
        ScaledLipschitzModule.__init__(self, factory)
        self.dim = dim
        self.momentum = momentum
        self.num_features = num_features
        self.centering = centering
        # register for saving the local mean on batch
        self.register_buffer("running_mean", torch.zeros((num_features,)))
        self.register_buffer("running_num_batches", torch.zeros((1,)))
        if bias:
            self.bias = nn.Parameter(torch.zeros((num_features,)), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.normalize = False
        if self.factory is not None:
            # register current_meansq for saving the local mean of square  on batch
            # register current_meansq for saving the  running mean of square on epoch
            self.factory.get_shared_buffer(self, "meansq", (num_features,))
            self.var_ones = torch.ones((num_features,))
            # registers for storing the total number samples in the epoch
            # Need two buffers to keep the total number when averging at the end of the epoch
            # the epoch average values will be considered as representing one batc only for the next epoch
            self.register_buffer("running_mean_sample_per_batches", torch.zeros((1,)))
            self.register_buffer("total_num_samples", torch.zeros((1,)))
            self.normalize = True
        self.eps = eps
        self.first = True
        self.local_num_elements = None
        self.current_mean = None

    # Reset the running statistics
    def reset_states(self):
        self.running_mean.zero_()
        self.running_num_batches.zero_()
        if self.normalize:
            # self.running_var.zero_()
            self.running_meansq.zero_()
            self.running_mean_sample_per_batches.zero_()
            self.total_num_samples.zero_()
            self.var_ones = self.var_ones.to(self.running_mean.device)

    # compute average of running values
    # divide running_mean and running_meansq by running_num_batches
    # keep the total_num_samples for rectified variance
    # update running_mean_sample_per_batches as the average number of samples per batch
    # update running_num_batches to 1.0 (as a batch tha represents theprevious epochs -to forget while learning)
    def update_running_values(self):
        if self.running_num_batches > 1:
            self.running_mean = self.running_mean / self.running_num_batches
            if self.normalize:
                self.running_meansq = self.running_meansq / self.running_num_batches
                self.total_num_samples = self.running_mean_sample_per_batches
                self.running_mean_sample_per_batches = (
                    self.running_mean_sample_per_batches / self.running_num_batches
                )
            self.running_num_batches = self.running_num_batches.zero_() + 1.0

    # retrieve the running mean in eval mode
    # if update is True, update the running values
    def get_running_mean(self, training=False, update=True):
        assert training == False, "Only in eval mode"
        # case asking for running mean before a step
        if self.running_num_batches == 0:
            return torch.zeros(self.running_mean.shape).to(self.running_mean.device)
        if update and (self.running_num_batches > 1):
            self.update_running_values()
        else:
            if self.normalize:
                if self.total_num_samples == 0:
                    self.total_num_samples = self.running_mean_sample_per_batches
        return self.running_mean / self.running_num_batches

    def forward(self, x):
        if self.dim is None:  # (0,2,3) for 4D tensor; (0,) for 2D tensor
            self.dim = (0,) + tuple(range(2, len(x.shape)))
        mean_shape = (1, -1) + (1,) * (len(x.shape) - 2)
        self.local_num_elements = x[:, 0].numel()
        if self.training:
            # on first batch initalize variables
            if self.first:
                self.reset_states()
                self.first = False
            # compute local mean (on batch of a single GPU)
            if self.centering:
                self.current_mean = x.mean(dim=self.dim)
            else:
                self.current_mean = torch.zeros((self.num_features,)).to(x.device)
            agrregated_mean = self.current_mean.clone()
            # compute local mean square (on batch of a single GPU)
            if self.normalize:
                xsq = x * x
                self.current_meansq = xsq.mean(dim=self.dim)
                current_meansq = self.current_meansq.clone().detach()

            # Get a tensor on the GPU for num_batches (aggregated on GPUs)
            # on a single GPU this value is always 1
            num_batches = self.running_num_batches.clone().detach().zero_() + 1.0

            # for multiGPU aggregate mean, mean square and num_batches
            if dist.is_initialized():
                dist.all_reduce(agrregated_mean.detach(), op=dist.ReduceOp.SUM)
                dist.all_reduce(num_batches.detach(), op=dist.ReduceOp.SUM)
                if self.normalize:
                    dist.all_reduce(current_meansq.detach(), op=dist.ReduceOp.SUM)

            # Accumulate running mean, mean square and num elements over the epoch
            # use aggregated mean and mean square for multi GPU
            with torch.no_grad():
                self.running_mean += agrregated_mean
                self.running_num_batches += num_batches
                if self.normalize:
                    self.running_meansq += current_meansq
                    self.running_mean_sample_per_batches += (
                        self.local_num_elements * num_batches
                    )
        else:
            self.current_mean = self.get_running_mean(self.training)

        # Compute scaling factor max(sqrt(variance)) either local (training) or global (eval)
        if self.normalize:
            scaling_norm = self.get_scaling_factor(self.training).to(x.device)
        else:
            scaling_norm = torch.ones((1,)).to(x.device)

        if self.bias is not None:
            return (
                x - self.current_mean.view(mean_shape)
            ) / scaling_norm + self.bias.view(mean_shape)
        else:
            return (x - self.current_mean.view(mean_shape)) / scaling_norm

    def vanilla_export(self, lambda_cumul):
        lambda_v = self.get_scaling_factor(False)
        size = self.running_mean.shape[0]
        bias = (
            -self.running_mean.detach()
            * lambda_cumul
            / self.running_num_batches.detach()
        )
        lambda_cumul *= lambda_v
        if self.bias is not None:
            bias += self.bias.detach() * lambda_cumul

        layer = ScaleBiasLayer(scalar=1.0, bias=True, size=size)
        layer.bias.data = bias
        return layer, lambda_cumul


class LipFactor(nn.Module, ScaledLipschitzModule):
    def __init__(
        self,
        factory: Optional[SharedLipFactory] = None,
    ):
        nn.Module.__init__(self)
        ScaledLipschitzModule.__init__(self, factory)

    def get_scaling_factor(self, training: bool):
        return self.factory.get_current_product_value(training)

    def forward(self, x):
        factor = self.get_scaling_factor(self.training)
        return x * factor

    def vanilla_export(self, lambda_cumul):
        factor = self.get_scaling_factor(False) / lambda_cumul
        if torch.abs(factor - 1.0) < 1e-6:
            return nn.Identity(), factor
        else:
            return ScaleBiasLayer(factor=factor, bias=False), factor


class BnLipSequential(TorchSequential):
    def __init__(self, lipFactory=None, layers=[]):
        super(BnLipSequential, self).__init__(*layers)
        self.lipFactory = lipFactory
        self.lfc = LipFactor(self.lipFactory)

    def update_running_values(self):
        for ll in self:
            if isinstance(ll, BatchLipNorm):
                ll.update_running_values()

    def forward(self, x):
        x = super(BnLipSequential, self).forward(x)
        x = self.lfc(x)
        return x

    def vanilla_export_layer(self, layer, lambda_cumul):
        if isinstance(layer, ScaledLipschitzModule):
            return layer.vanilla_export(lambda_cumul)
        return copy.deepcopy(layer), lambda_cumul

    def vanilla_export(self):
        lambda_cumul = 1.0
        layers = []
        for nn, ll in self.named_modules():
            layer, lambda_cumul = self.vanilla_export_layer(ll, lambda_cumul)
            layers.append((nn, layer))
        layer, lambda_cumul = self.vanilla_export_layer(self.lfc, lambda_cumul)
        if not isinstance(layer, nn.Identity):
            layers.append((f"lfc", layer))
        assert torch.abs(lambda_cumul - 1.0) < 1e-5, "Lipschitz constant is not one"
        return TorchSequential(OrderedDict(layers)).eval()
