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

    #compute average of running values
    def update_running_values(self):
        if self.running_num_batches>1:
            self.running_mean = self.running_mean/self.running_num_batches
            self.running_num_batches = self.running_num_batches.zero_()+1.0

    def get_running_mean(self,training=False):
        assert training == False, "Only in eval mode"
        # case asking for running mean before a step
        if self.running_num_batches == 0: 
            return torch.zeros(self.running_mean.shape).to(self.running_mean.device)
        if self.running_num_batches > 1:
            self.update_running_values()
        return self.running_mean/self.running_num_batches
    
    def forward(self, x):
        if self.dim is None:  # (0,2,3) for 4D tensor; (0,) for 2D tensor
            self.dim = (0,) + tuple(range(2, len(x.shape)))
        mean_shape = (1, -1) + (1,) * (len(x.shape) - 2)
        if self.training:            
            if self.first:
                self.reset_states()
                self.first = False
            mean = x.mean(dim=self.dim)
            with torch.no_grad():
                self.running_mean += mean
                self.running_num_batches += 1.0

            if dist.is_initialized():
                dist.all_reduce(self.running_mean.detach(), op=dist.ReduceOp.SUM)
                dist.all_reduce(self.running_num_batches.detach(), op=dist.ReduceOp.SUM)
                self.running_mean /= dist.get_world_size()
        else:
            mean = self.get_running_mean(self.training)
        if self.bias is not None:
            return x - mean.view(mean_shape) + self.bias.view(mean_shape)
        else:
            return x - mean.view(mean_shape)

BatchCentering2D = BatchCentering

class SharedLipFactory:
    def __init__(self, 
            eps: float = 1e-5
        ):
        self.buffers_name2module = {}
        self.eps = eps

    def get_shared_buffer(self, module, buffer_name, shape = (1,)):
        """Retrieve or create a shared buffer within the model."""
        if buffer_name not in self.buffers_name2module:
            self.buffers_name2module[buffer_name] = []
        buffer = torch.ones(shape)
        module.register_buffer("running_"+buffer_name, buffer)
        current_buffer = torch.ones(shape)
        module.register_buffer("current_"+buffer_name, current_buffer)
        self.buffers_name2module[buffer_name].append(module)

    def get_var_value(self,module,training, current_mean = None, currant_sample_num = None):
        """Retrieve the current value of a module shared buffers."""
        assert len(self.buffers_name2module) == 1, "Only one buffer type is supported"
        buffer_name = list(self.buffers_name2module.keys())[0]
        if training:
            var_sum = getattr(module,"current_"+buffer_name)
            mean_sum = current_mean #getattr(module,"current_mean")
            assert mean_sum is not None, "current_mean should be provided in training mode"
            assert currant_sample_num is not None, "currant_sample_num should be provided in training mode" 
            var_factor = currant_sample_num/(currant_sample_num -1)
            #print(int(os.environ["LOCAL_RANK"])," training : current_"+buffer_name,var_sum)
            num_batches = 1.0
        else:
            mean_sum = module.get_running_mean() #update variables in test mode
            var_sum = getattr(module,"running_"+buffer_name)
            num_batches = getattr(module,"running_num_batches")
            #mean_sample_per_batches = getattr(module,"running_mean_sample_per_batches")
            #print(mean_sample_per_batches, num_batches)
            total_num_samples = getattr(module,"total_num_samples")
            var_factor = total_num_samples/(total_num_samples-1)
            #var_factor = mean_sample_per_batches*num_batches/(mean_sample_per_batches*num_batches-1)
            #mean_factor = mean_sample_per_batches*num_batches/(mean_sample_per_batches*num_batches-1)
            #print(int(os.environ["LOCAL_RANK"])," test : current_"+buffer_name,var_sum)
        if num_batches == 0:
            return torch.ones((1,))
        var = (var_sum - (mean_sum*mean_sum))*var_factor
        print("var" , var)
        var  = torch.where(var < self.eps, 
                                        torch.ones(var.shape).to(var.device), 
                                        var)
        return var.max()
    
    def get_current_product_value(self,training):
        """Retrieve the current product of the shared buffers."""
        assert len(self.buffers_name2module) == 1, "Only one buffer type is supported"
        buffer_name = list(self.buffers_name2module.keys())[0]
        buffers = [module.get_scaling_factor(training) for module in self.buffers_name2module[buffer_name]]
        return torch.prod(torch.stack(buffers))

class ScaledLipschitzModule(abc.ABC):
    """
    This class allow to set learnable lipschitz parameter of a layer. 
    
    """ 

    def __init__(self, factory: Optional[SharedLipFactory] = None, factor_name: str = "var"):
        
        # Factory of factors
        self.factory = factory
        # factor name:
        self.factor_name = factor_name
    def get_scaling_factor(self, training: bool= False, current_mean = None, currant_sample_num = None):
        var = self.get_variance_factor(training, current_mean, currant_sample_num)
        return var.sqrt()
    def get_variance_factor(self, training: bool, current_mean = None, currant_sample_num = None):
        if self.factory is None:
            return torch.ones((1,))
        else:
            return self.factory.get_var_value(self,training, current_mean, currant_sample_num)

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
        bias: if `True`, adds a learnable bias to the output
        of shape (size,). Default: `True`

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
        self.register_buffer("running_mean", torch.zeros((num_features,)))
        self.register_buffer("running_num_batches", torch.zeros((1,)))
        if bias:
            self.bias = nn.Parameter(torch.zeros((num_features,)), requires_grad=True)
        else:
            self.register_parameter("bias", None)
        self.normalize = False
        if self.factory is not None:
            #self.factory.get_shared_buffer(self, "var", (num_features,))
            self.factory.get_shared_buffer(self, "meansq", (num_features,))
            self.var_ones = torch.ones((num_features,))
            self.register_buffer("running_mean_sample_per_batches", torch.zeros((1,)))
            self.register_buffer("total_num_samples", torch.zeros((1,)))
            self.normalize = True
        self.eps = eps
        self.first = True

    def reset_states(self):
        self.running_mean.zero_()
        self.running_num_batches.zero_()
        if self.normalize:
            #self.running_var.zero_()
            self.running_meansq.zero_()
            self.running_mean_sample_per_batches.zero_()
            self.total_num_samples.zero_()
            self.var_ones = self.var_ones.to(self.running_mean.device)

    #compute average of running values
    def update_running_values(self):
        if self.running_num_batches>1:
            self.running_mean = self.running_mean/self.running_num_batches        
            if self.normalize:
                #self.running_var = self.running_var/self.running_num_batches
                self.running_meansq = self.running_meansq/self.running_num_batches
                self.total_num_samples = self.running_mean_sample_per_batches 
                self.running_mean_sample_per_batches = self.running_mean_sample_per_batches/self.running_num_batches
            self.running_num_batches = self.running_num_batches.zero_()+1.0

    def get_running_mean(self,training=False):
        assert training == False, "Only in eval mode"
        # case asking for running mean before a step
        if self.running_num_batches == 0: 
            return torch.zeros(self.running_mean.shape).to(self.running_mean.device)
        if self.running_num_batches > 1:
            self.update_running_values()
        else:    
            if self.normalize:
                if self.total_num_samples == 0:
                    self.total_num_samples = self.running_mean_sample_per_batches
        return self.running_mean/self.running_num_batches
    
    def forward(self, x):
        if self.dim is None:  # (0,2,3) for 4D tensor; (0,) for 2D tensor
            self.dim = (0,) + tuple(range(2, len(x.shape)))
        mean_shape = (1, -1) + (1,) * (len(x.shape) - 2)
        num_elements = x[:,0].numel()
        if self.training:
            #print(int(os.environ["LOCAL_RANK"])," input " ,x)
            if self.first:
                self.reset_states()
                self.first = False
            if self.centering:
                mean = x.mean(dim=self.dim)
            else:
                mean = torch.zeros((self.num_features,)).to(x.device)
            if self.normalize:
                xsq = x*x
                current_meansq = xsq.mean(dim=self.dim)
                test_var = x.var(dim=self.dim)
                '''
                print(x)
                print(mean)
                print(mean*mean)
                print(test_var,(current_meansq - mean*mean)*num_elements/(num_elements-1))
                print(test_var - (current_meansq - mean*mean)*num_elements/(num_elements-1))'''
                #current_var = x.var(dim=self.dim)
                # constant case don't divide by zero
                '''current_var = torch.where(current_var < self.eps, 
                                        self.var_ones, 
                                        current_var)'''
            with torch.no_grad():
                self.running_mean += mean
                self.running_num_batches += 1.0
                if self.normalize:
                    self.current_meansq = current_meansq #  in training use the current lambda
                    #self.current_var = current_var #  in training use the current lambda
                    self.running_meansq += self.current_meansq
                    #self.running_var += self.current_var
                    self.running_mean_sample_per_batches += num_elements #x.shape[0]
            if dist.is_initialized():
                print("dist training", self.running_mean.shape, self.running_num_batches.shape)
                print("dist training", self.running_mean, self.running_num_batches)
                dist.all_reduce(self.running_mean.detach(), op=dist.ReduceOp.SUM)
                dist.all_reduce(self.running_num_batches.detach(), op=dist.ReduceOp.SUM)
                print("dist reduced", self.running_mean, self.running_num_batches)
                #divison by world size included in num_batches count
                #self.running_mean /= dist.get_world_size()
                if self.normalize:
                    #dist.all_reduce(self.running_var.detach(), op=dist.ReduceOp.SUM)
                    dist.all_reduce(self.running_meansq.detach(), op=dist.ReduceOp.SUM)
                    #divison by world size included in num_batches count
                    #dist.all_reduce(self.current_var.detach(), op=dist.ReduceOp.SUM)
                    #self.current_var /= dist.get_world_size()
            #print("training mean", mean_shape, mean.view(mean_shape).flatten().float().detach().cpu().numpy()[0:3],self.get_running_mean(False).flatten().detach().cpu().numpy()[0:3], self.running_num_batches.cpu().numpy(),)
            #print("training var", self.current_var.shape, self.current_var.flatten()[0:3].detach().cpu().numpy(),self.current_var.flatten().max().detach().cpu().numpy(),(self.running_var.flatten()[0:3]/self.running_num_batches).detach().cpu().numpy())
        else:
            mean = self.get_running_mean(self.training)
            #print("test mean", mean_shape, mean.view(mean_shape).flatten().detach().cpu().numpy()[0:3],self.running_num_batches.cpu().numpy())
            #print("test var", self.running_var.shape, (self.running_var.flatten()[0:3]/self.running_num_batches).detach().cpu().numpy(),(self.running_var.flatten().max()/self.running_num_batches).detach().cpu().numpy())
        if self.normalize:
            scaling_norm = self.get_scaling_factor(self.training, mean, num_elements).to(x.device)
        else:
            scaling_norm = torch.ones((1,)).to(x.device)
            #self.running_var/self.running_num_batches # in eval use running lambda
        #if lambda_max < self.eps:
        #    lambda_max = 1.0
        if self.bias is not None:
            return (x - mean.view(mean_shape))/scaling_norm + self.bias.view(mean_shape)
        else:
            return (x - mean.view(mean_shape))/scaling_norm
        
    def vanilla_export(self, lambda_cumul):
        lambda_v = self.get_scaling_factor(False)
        size = self.running_mean.shape[0]
        bias = -self.running_mean.detach()*lambda_cumul/self.running_num_batches.detach()
        lambda_cumul *= lambda_v   
        if self.bias is not None:
            bias += self.bias.detach()*lambda_cumul
        
        layer = ScaleBiasLayer(scalar=1.0,bias=True, size=size)
        layer.bias.data = bias
        return layer, lambda_cumul



class LipFactor(nn.Module,ScaledLipschitzModule):
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
        return x*factor
    def vanilla_export(self,lambda_cumul):
        factor = self.get_scaling_factor(False)/lambda_cumul
        if torch.abs(factor - 1.0)< 1e-6:
            return nn.Identity(), factor
        else:
            return ScaleBiasLayer(factor=factor,bias=False), factor


class BnLipSequential(TorchSequential):
    def __init__(self, lipFactory = None, layers = []):
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
    '''def forward_and_keep(self, x):
        getl_all_layers = []
        getl_all_layers.append(x.detach().cpu().numpy())
        for i in range(self.num_layers):
            x = self.layers[i](x)
            getl_all_layers.append(x.detach().cpu().numpy())
            x = self.normalizations[i](x)
            getl_all_layers.append(x.detach().cpu().numpy())
            if i < self.num_layers-1:
                x = self.activation(x)  # No activation on the last layer   
            getl_all_layers.append(x.detach().cpu().numpy())
        x = self.fc(x)
        getl_all_layers.append(x.detach().cpu().numpy())
        x = self.lfc(x)
        getl_all_layers.append(x.detach().cpu().numpy())
        return x, getl_all_layers'''
    
    def vanilla_export_layer(self, layer, lambda_cumul):
        '''if isinstance(layer, LipschitzModule):
            layer = layer.vanilla_export()
            if hasattr(layer, "bias") and layer.bias is not None:
                layer.bias.data = layer.bias.data*lambda_cumul
            return layer, lambda_cumul'''
        if isinstance(layer, ScaledLipschitzModule):
            return layer.vanilla_export(lambda_cumul)
        return copy.deepcopy(layer), lambda_cumul
    
    def vanilla_export(self):
        lambda_cumul = 1.0
        layers = []
        for nn, ll in self.named_modules():
            layer, lambda_cumul = self.vanilla_export_layer(ll, lambda_cumul)
            layers.append((nn,layer))
        layer, lambda_cumul = self.vanilla_export_layer(self.lfc, lambda_cumul)
        if not isinstance(layer, nn.Identity):
            layers.append((f"lfc",layer))
        assert torch.abs(lambda_cumul - 1.0)< 1e-5, "Lipschitz constant is not one"
        return TorchSequential(OrderedDict(layers)).eval()
