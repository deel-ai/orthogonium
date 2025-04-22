import torch
from torch import nn as nn


class ConcatResidual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.add_module("fn", fn)

    def forward(self, x):
        # split x
        x1, x2 = x.chunk(2, dim=1)
        # apply function
        out = self.fn(x2)
        # concat and return
        return torch.cat([x1, out], dim=1)


class L2NormResidual(nn.Module):
    def __init__(self, fn, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.add_module("fn", fn)

    def forward(self, x):
        # apply function
        out = self.fn(x)
        # concat and return
        return torch.sqrt(0.5 * x**2 + 0.5 * out**2 + self.eps)


class AdditiveResidual(nn.Module):
    def __init__(self, fn, init_val=1.0):
        super().__init__()
        self.add_module("fn", fn)
        self.alpha = nn.Parameter(torch.tensor(init_val), requires_grad=True)

    def forward(self, x):
        # apply function
        out = self.fn(x)
        # alpha = self.alpha.clamp(0, 1)
        alpha = torch.sigmoid(self.alpha)  # check if alpha don't grow to infinity
        return alpha * x + (1 - alpha) * out


class PrescaledAdditiveResidual(nn.Module):
    def __init__(self, fn, init_val=1.0):
        super().__init__()
        self.add_module("fn", fn)
        self.alpha = nn.Parameter(torch.tensor(init_val), requires_grad=True)

    def forward(self, x):
        # apply function
        out = self.fn(x * self.alpha)
        lip_cst = 1.0 + torch.abs(self.alpha)
        # we divide by lip const on each branch as it is more numerically stable
        return x / lip_cst + (1.0 / lip_cst) * out
