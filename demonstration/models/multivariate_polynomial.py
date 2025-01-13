""" Multivariate polynomial model """
import torch
import torch.nn
from models.legendre_polynomial import Legendre

# pylint: disable=E1101


class MultiVariatePoly(torch.nn.Module):
    """Class for the multivariate polynomial module."""

    def __init__(self, dim, order):
        super(MultiVariatePoly, self).__init__()
        self.order = order
        self.dim = dim
        self.polys = Legendre(order)
        self.num = (order + 1)**dim
        self.linear = torch.nn.Linear(self.num, 1)

    def forward(self, x):
        """Compute the forward pass of the multivariate polynomial module."""
        poly_eval = []
        leg_eval = torch.cat([
            self.polys(x[:, i]).reshape(1, x.shape[0], self.order + 1)
            for i in range(self.dim)
        ])
        for i in range(x.shape[0]):
            poly_eval.append(
                torch.cartesian_prod(*leg_eval[:,
                                               i, :]).prod(dim=1).view(1, -1))
        poly_eval = torch.cat(poly_eval)
        return self.linear(poly_eval)
