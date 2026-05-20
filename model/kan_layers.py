"""Kolmogorov-Arnold Network (KAN) Layers.

KAN replaces fixed activations with learnable univariate spline functions.
Reference: Liu et al., "KAN: Kolmogorov-Arnold Networks", 2024.

Instead of y = activation(Wx + b), KAN computes:
    y_j = sum_i( spline_{ij}(x_i) )

Each (input, output) pair has its own learnable B-spline function.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """Kolmogorov-Arnold Network Linear Layer.

    Replaces standard linear layer with learnable B-spline transformations.
    Each input-output connection has a learnable spline function.

    Args:
        in_features: Input dimension
        out_features: Output dimension
        grid_size: Number of B-spline grid intervals (default: 5)
        spline_order: B-spline order k (default: 3 for cubic)
        scale_noise: Initialization noise scale for spline weights
        scale_base: Scale for base weight initialization
        scale_spline: Scale for spline weight initialization
        base_activation: Activation for residual path (default: SiLU)
        grid_eps: Small offset for grid extension
        grid_range: [min, max] range for input normalization
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: type = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: list = [-1.0, 1.0],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Grid setup: extended grid for B-spline boundary conditions
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer("grid", grid)

        # Number of spline basis functions
        self.num_bases = grid_size + spline_order

        # Spline coefficients: (out_features, in_features, num_bases)
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, self.num_bases)
            * scale_spline
            * scale_noise
            / math.sqrt(in_features)
        )

        # Base weight for residual connection: (out_features, in_features)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features)
            * scale_base
            / math.sqrt(in_features)
        )

        # Base activation
        self.base_activation = base_activation() if base_activation else nn.Identity()

        # Learnable scaling per output
        self.spline_scaler = nn.Parameter(torch.ones(out_features, in_features))

    def b_spline_basis(self, x: torch.Tensor) -> torch.Tensor:
        """Compute B-spline basis functions using Cox-de Boor recursion.

        Args:
            x: (batch, in_features) input values

        Returns:
            bases: (batch, in_features, num_bases) B-spline basis values
        """
        # x: (B, in) -> (B, in, 1) for broadcasting
        x = x.unsqueeze(-1)
        grid = self.grid  # (G,) where G = grid_size + 2*spline_order + 1

        # Order 0: piecewise constant basis
        # bases[i] = 1 if grid[i] <= x < grid[i+1], else 0
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

        # Cox-de Boor recursion for higher orders
        for k in range(1, self.spline_order + 1):
            # Left term: (x - t_i) / (t_{i+k} - t_i) * B_{i,k-1}(x)
            denom_left = grid[k:-1] - grid[:-k-1]
            denom_left = denom_left.clamp(min=1e-8)
            left = (x - grid[:-k-1]) / denom_left * bases[..., :-1]

            # Right term: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
            denom_right = grid[k+1:] - grid[1:-k]
            denom_right = denom_right.clamp(min=1e-8)
            right = (grid[k+1:] - x) / denom_right * bases[..., 1:]

            bases = left + right

        return bases  # (B, in, num_bases)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (*batch_dims, in_features) input tensor

        Returns:
            y: (*batch_dims, out_features) output tensor
        """
        # Store original shape for reshaping at end
        original_shape = x.shape[:-1]
        x = x.reshape(-1, self.in_features)  # (B, in)
        batch_size = x.shape[0]

        # Compute B-spline basis: (B, in, num_bases)
        bases = self.b_spline_basis(x)

        # Spline output: sum over bases, scaled by spline_weight
        # spline_weight: (out, in, num_bases)
        # spline_scaler: (out, in)
        # bases: (B, in, num_bases)
        # Want: (B, out) = sum over in and num_bases
        # Scale weights, not bases
        scaled_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)  # (out, in, num_bases)
        spline_out = torch.einsum("bin,oin->bo", bases, scaled_weight)

        # Base path: activation -> linear
        base_out = F.linear(self.base_activation(x), self.base_weight)

        # Combine
        y = spline_out + base_out

        # Restore batch dimensions
        return y.reshape(*original_shape, self.out_features)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """Compute regularization losses for spline smoothness.

        Args:
            regularize_activation: Weight for L1 activation regularization
            regularize_entropy: Weight for entropy regularization

        Returns:
            reg_loss: Scalar regularization loss
        """
        # L1 on spline weights (encourages sparsity)
        l1_reg = self.spline_weight.abs().mean()

        # Smoothness: penalize second derivative (approximated by weight differences)
        if self.num_bases >= 3:
            d2 = self.spline_weight[..., 2:] - 2 * self.spline_weight[..., 1:-1] + self.spline_weight[..., :-2]
            smooth_reg = (d2 ** 2).mean()
        else:
            smooth_reg = torch.tensor(0.0, device=self.spline_weight.device)

        return regularize_activation * l1_reg + regularize_entropy * smooth_reg


class KANLayer(nn.Module):
    """Full KAN layer with optional normalization and dropout."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.kan_linear = KANLinear(
            in_features, out_features,
            grid_size=grid_size, spline_order=spline_order
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layernorm = nn.LayerNorm(out_features) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layernorm(self.dropout(self.kan_linear(x)))

    def regularization_loss(self, **kwargs):
        return self.kan_linear.regularization_loss(**kwargs)
