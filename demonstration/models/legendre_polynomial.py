""" Univariate Legendre Polynomial """
import torch
import torch.nn

# pylint: disable=E1101


class Legendre(torch.nn.Module):
    """Class for the Legendre polynomial module."""

    def __init__(self, PolyDegree):
        """
        Initialize the Legendre polynomial module.

        Args:
            PolyDegree (int): The degree of the Legendre polynomial.
        """
        super().__init__()
        self.degree = PolyDegree

    def legendre(self, x, degree):
        """
        Compute the Legendre polynomial of the given degree.

        Args:
            x (torch.Tensor): The input tensor.
            degree (int): The degree of the Legendre polynomial.

        Returns:
            torch.Tensor: The Legendre polynomial evaluated at x.
        """
        x = x.reshape(-1, 1)
        list_poly = []
        zeroth_pol = torch.ones(x.size(0), 1)
        list_poly.append(zeroth_pol)
        # retvar[:, 0] = x * 0 + 1
        if degree > 0:
            first_pol = x
            list_poly.append(first_pol)
            ith_pol = torch.clone(first_pol)
            ith_m_pol = torch.clone(zeroth_pol)

            for ii in range(1, degree):
                ith_p_pol = (
                    (2 * ii + 1) * x * ith_pol - ii * ith_m_pol) / (ii + 1)
                list_poly.append(ith_p_pol)
                ith_m_pol = torch.clone(ith_pol)
                ith_pol = torch.clone(ith_p_pol)
        list_poly = torch.cat(list_poly, 1)
        return list_poly

    def forward(self, x):
        """
        Compute the forward pass of the Legendre polynomial module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The Legendre polynomial evaluated at x.
        """
        eval_poly = self.legendre(x, self.degree)
        return eval_poly
