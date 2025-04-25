import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


SQRT_2 = np.sqrt(2)


class Abs(nn.Module):
    def __init__(self):
        """
        Initializes an instance of the Abs class.

        This method is automatically called when a new object of the Abs class
        is instantiated. It calls the initializer of its superclass to ensure
        proper initialization of inherited class functionality, setting up
        the required base structures or attributes.
        """
        super(Abs, self).__init__()

    def forward(self, z):
        return torch.abs(z)


class SoftHuber(nn.Module):
    def __init__(self, delta=0.05):
        """
        Initializes the SoftHuber class.
        This class implements the Soft Huber loss function, which is a
        differentiable approximation of the Huber loss. The Soft Huber loss
        behaves like abs(x) when the absolute error is large and like x**2
        when the absolute error is small. The transition between these two
        behaviors is controlled by the delta parameter.

        Parameters:
            delta (float): The threshold at which to switch between L1 and L2 loss.
        """
        super(SoftHuber, self).__init__()
        self.delta = delta

    def forward(self, z):
        # we dont multiply by delta**2 in order to have a Lipschitz constant of 1
        return self.delta * (torch.sqrt(1 + (z / self.delta) ** 2) - 1)


class MaxMin(nn.Module):
    def __init__(self, axis=1):
        """
        This class implements the MaxMin activation function. Which is a
        pairwise activation function that returns the maximum and minimum (ordered)
        of each pair of elements in the input tensor.

        Parameters
            axis : int, default=1 the axis along which to apply the activation function.

        """
        self.axis = axis
        super(MaxMin, self).__init__()

    def forward(self, z):
        a, b = z.split(z.shape[self.axis] // 2, self.axis)
        c, d = torch.max(a, b), torch.min(a, b)
        return torch.cat([c, d], dim=self.axis)


class HouseHolder(nn.Module):
    def __init__(self, channels, axis=1):
        """
        A activation that applies a parameterized transformation via Householder
        reflection technique. It is initialized with the number of input channels, which must
        be even, and an axis that determines the dimension along which operations are applied.
        This is a corrected version of the original implementation from Singla et al. (2019),
        which features a 1/sqrt(2) scaling factor to be 1-Lipschitz.

        Attributes:
            theta (torch.nn.Parameter): Learnable parameter that determines the transformation
                applied via Householder reflection.
            axis (int): Dimension along which the operation is performed.

        Args:
            channels (int): Total number of input channels. Must be an even number.
            axis (int): Dimension along which the transformation is applied. Default is 1.
        """
        super(HouseHolder, self).__init__()
        assert (channels % 2) == 0
        eff_channels = channels // 2

        self.theta = nn.Parameter(
            0.5 * np.pi * torch.ones(1, eff_channels, 1, 1), requires_grad=True
        )
        self.axis = axis

    def forward(self, z):
        theta = self.theta
        x, y = z.split(z.shape[self.axis] // 2, self.axis)

        selector = (x * torch.sin(0.5 * theta)) - (y * torch.cos(0.5 * theta))

        a_2 = x * torch.cos(theta) + y * torch.sin(theta)
        b_2 = x * torch.sin(theta) - y * torch.cos(theta)

        a = x * (selector <= 0) + a_2 * (selector > 0)
        b = y * (selector <= 0) + b_2 * (selector > 0)
        return torch.cat([a, b], dim=self.axis) / SQRT_2


class HouseHolder_Order_2(nn.Module):
    def __init__(self, channels, axis=1):
        """
        Represents a layer or module that performs operations using Householder
        transformations of order 2, parameterized by angles corresponding to
        each group of channels. This is a corrected version of the original
        implementation from Singla et al. (2019), which features a 1/sqrt(2)
        scaling factor to be 1-Lipschitz.

        Attributes:
            num_groups (int): The number of groups, which is half the number
            of channels provided as input.

            axis (int): The axis along which the computation is performed.

            theta0 (torch.nn.Parameter): A tensor parameter of shape `(num_groups,)`
            representing the first set of angles (in radians) used in the
            parameterization.

            theta1 (torch.nn.Parameter): A tensor parameter of shape `(num_groups,)`
            representing the second set of angles (in radians) used in the
            parameterization.

            theta2 (torch.nn.Parameter): A tensor parameter of shape `(num_groups,)`
            representing the third set of angles (in radians) used in the
            parameterization.

        Args:
            channels (int): The total number of input channels. Must be an even
            number, as it will be split into groups.

            axis (int, optional): Specifies the axis for computations. Defaults
            to 1.
        """
        super(HouseHolder_Order_2, self).__init__()
        assert (channels % 2) == 0
        self.num_groups = channels // 2
        self.axis = axis

        self.theta0 = nn.Parameter(
            (np.pi * torch.rand(self.num_groups)), requires_grad=True
        )
        self.theta1 = nn.Parameter(
            (np.pi * torch.rand(self.num_groups)), requires_grad=True
        )
        self.theta2 = nn.Parameter(
            (np.pi * torch.rand(self.num_groups)), requires_grad=True
        )

    def forward(self, z):
        theta0 = torch.clamp(self.theta0.view(1, -1, 1, 1), 0.0, 2 * np.pi)

        x, y = z.split(z.shape[self.axis] // 2, self.axis)
        z_theta = (torch.atan2(y, x) - (0.5 * theta0)) % (2 * np.pi)

        theta1 = torch.clamp(self.theta1.view(1, -1, 1, 1), 0.0, 2 * np.pi)
        theta2 = torch.clamp(self.theta2.view(1, -1, 1, 1), 0.0, 2 * np.pi)
        theta3 = 2 * np.pi - theta1
        theta4 = 2 * np.pi - theta2

        ang1 = 0.5 * (theta1)
        ang2 = 0.5 * (theta1 + theta2)
        ang3 = 0.5 * (theta1 + theta2 + theta3)
        ang4 = 0.5 * (theta1 + theta2 + theta3 + theta4)

        select1 = torch.logical_and(z_theta >= 0, z_theta < ang1)
        select2 = torch.logical_and(z_theta >= ang1, z_theta < ang2)
        select3 = torch.logical_and(z_theta >= ang2, z_theta < ang3)
        select4 = torch.logical_and(z_theta >= ang3, z_theta < ang4)

        a1 = x
        b1 = y

        a2 = x * torch.cos(theta0 + theta1) + y * torch.sin(theta0 + theta1)
        b2 = x * torch.sin(theta0 + theta1) - y * torch.cos(theta0 + theta1)

        a3 = x * torch.cos(theta2) + y * torch.sin(theta2)
        b3 = -x * torch.sin(theta2) + y * torch.cos(theta2)

        a4 = x * torch.cos(theta0) + y * torch.sin(theta0)
        b4 = x * torch.sin(theta0) - y * torch.cos(theta0)

        a = (a1 * select1) + (a2 * select2) + (a3 * select3) + (a4 * select4)
        b = (b1 * select1) + (b2 * select2) + (b3 * select3) + (b4 * select4)

        z = torch.cat([a, b], dim=self.axis) / SQRT_2
        return z
