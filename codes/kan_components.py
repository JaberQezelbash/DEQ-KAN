# kan_components.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SplineActivation(nn.Module):
    """
    A simplified 1D 'spline-like' activation with Gaussian basis.
    """
    def __init__(self, num_knots=10):
        super(SplineActivation, self).__init__()
        self.num_knots = num_knots
        # Fixed grid from -3 to 3
        self.knots = nn.Parameter(torch.linspace(-3, 3, steps=num_knots), requires_grad=False)
        # Initialize coefficients randomly
        self.coeffs = nn.Parameter(0.01 * torch.randn(num_knots))
        self.bias   = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x_unsq = x.unsqueeze(-1)
        knot_unsq = self.knots.view(1, -1)
        diff = x_unsq - knot_unsq
        basis = torch.exp(-0.5 * diff**2)
        out = torch.sum(basis * self.coeffs, dim=-1) + self.bias
        return out

class KANLayer(nn.Module):
    """
    One 'layer' of a KAN without an internal skip.
    """
    def __init__(self, dim_in, dim_out, num_knots=10, dropout_p=0.0):
        super(KANLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = nn.Dropout(p=dropout_p)
        self.activations = nn.ModuleList([
            SplineActivation(num_knots=num_knots)
            for _ in range(dim_in * dim_out)
        ])

    def forward(self, z):
        z = self.dropout(z)
        bs = z.shape[0]
        out = z.new_zeros(bs, self.dim_out)
        idx = 0
        for j in range(self.dim_out):
            sum_j = 0
            for i in range(self.dim_in):
                spline_f = self.activations[idx]
                sum_j += spline_f(z[:, i])
                idx += 1
            out[:, j] = sum_j
        return out

class KANBlock(nn.Module):
    """
    A block with two KAN layers and a residual skip:
      z_next = z + KAN2(ReLU(KAN1(z))).
    """
    def __init__(self, dim_in, num_knots=10, dropout_p=0.2):
        super(KANBlock, self).__init__()
        self.kan1 = KANLayer(dim_in, dim_in, num_knots, dropout_p=dropout_p)
        self.kan2 = KANLayer(dim_in, dim_in, num_knots, dropout_p=dropout_p)

    def forward(self, z):
        tmp = self.kan1(z)
        tmp = F.relu(tmp)
        tmp = self.kan2(tmp)
        return z + tmp

if __name__ == "__main__":
    # Quick test for KANBlock
    x = torch.randn(2, 128)
    block = KANBlock(dim_in=128, num_knots=10, dropout_p=0.2)
    out = block(x)
    print("KANBlock output shape:", out.shape)
