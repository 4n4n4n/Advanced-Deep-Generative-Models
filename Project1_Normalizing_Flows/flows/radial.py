import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .nf_utils import Flow


class Radial(Flow):
    """Radial transformation.

    Args:
        dim: dimension of input/output data, int
    """

    def __init__(self, dim: int = 2):
        """Create and initialize an affine transformation."""
        super().__init__()

        self.dim = dim

        self.x0 = nn.Parameter(
            torch.Tensor(
                self.dim,
            )
        )  # Vector used to parametrize z_0
        self.pre_alpha = nn.Parameter(
            torch.Tensor(
                1,
            )
        )  # Scalar used to indirectly parametrized \alpha
        self.pre_beta = nn.Parameter(
            torch.Tensor(
                1,
            )
        )  # Scaler used to indireclty parametrized \beta

        stdv = 1.0 / math.sqrt(self.dim)
        self.pre_alpha.data.uniform_(-stdv, stdv)
        self.pre_beta.data.uniform_(-stdv, stdv)
        self.x0.data.uniform_(-stdv, stdv)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the forward transformation for the given input x.

        Args:
            x: input sample, shape [batch_size, dim]

        Returns:
            y: sample after forward transformation, shape [batch_size, dim]
            log_det_jac: log determinant of the jacobian of the forward transformation, shape [batch_size]
        """
        B, D = x.shape

        ##########################################################
        # YOUR CODE HERE
        dz = x - self.x0
        r = torch.norm(dz, dim = 1)
        assert(r.shape == (B,))
        spa = torch.log(1+torch.exp(self.pre_alpha))
        spb = torch.log(1+torch.exp(self.pre_beta)) - spa
        y = torch.zeros([B,D])
        log_det_jac = torch.zeros(B)
        denom = r+spa
        gr = spb /denom # (B,)
        y = x + gr.unsqueeze(-1) * dz  # (B,1) and (B,D) be careful of broadcasting
        log_det_jac = (D-1)*torch.log(gr + 1) + torch.log(1+gr-r* spb/denom**2)

        # for i in range(B):
        #     gri = (sp(self.pre_beta)/(sp(self.pre_alpha)+r[i]))
        #     gri_ = -sp(self.pre_beta)/((sp(self.pre_alpha)+r[i])*(sp(self.pre_alpha)+r[i]))
        #     y[i] = x[i] + gri *dz[i]
        #     log_det_jac[i] = (D-1)*torch.log(1+gri) + torch.log(1+gri + gri_*r[i])

        ##########################################################

        assert y.shape == (B, D)
        assert log_det_jac.shape == (B,)

        return y, log_det_jac

    def inverse(self, y: Tensor) -> None:
        """Compute the inverse transformation given an input y.

        Args:
            y: input sample. shape [batch_size, dim]

        Returns:
            x: sample after inverse transformation. shape [batch_size, dim]
            inv_log_det_jac: log determinant of the jacobian of the inverse transformation, shape [batch_size]
        """
        raise ValueError("The inverse transformation is not known in closed form.")
