"""Hyperbolic Geometry for Binding Interactions.

Implements the Lorentz (hyperboloid) model of hyperbolic space.
Hyperbolic space has exponentially growing volume with distance,
making it natural for hierarchical binding relationships.

References:
- Nickel & Kiela, "Poincare Embeddings for Learning Hierarchical Representations", NeurIPS 2017
- Ganea et al., "Hyperbolic Neural Networks", NeurIPS 2018
- Chami et al., "Hyperbolic Graph Convolutional Neural Networks", NeurIPS 2019
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LorentzManifold:
    """Lorentz (hyperboloid) model of hyperbolic space.

    Points lie on the hyperboloid: -x_0^2 + x_1^2 + ... + x_d^2 = -1/c
    where c > 0 is the curvature parameter.

    The Lorentz model is numerically more stable than the Poincare ball
    for high-dimensional embeddings.

    Args:
        curvature: Manifold curvature c > 0 (default: 1.0)
        eps: Small constant for numerical stability
    """

    def __init__(self, curvature: float = 1.0, eps: float = 1e-6):
        self.c = curvature
        self.eps = eps
        self.sqrt_c = math.sqrt(curvature)

    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Minkowski (Lorentz) inner product.

        <x, y>_L = -x_0 * y_0 + x_1 * y_1 + ... + x_d * y_d

        Args:
            x, y: (..., d+1) tensors on hyperboloid

        Returns:
            inner: (..., 1) Minkowski inner products
        """
        # Time component (index 0) has negative sign
        time_inner = -x[..., 0:1] * y[..., 0:1]
        space_inner = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
        return time_inner + space_inner

    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Minkowski norm: sqrt(|<x, x>_L|)."""
        inner = self.minkowski_inner(x, x)
        return torch.sqrt(torch.clamp(inner.abs(), min=self.eps))

    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project Euclidean point onto hyperboloid.

        Given spatial components x[1:], compute time component x[0] such that
        -x_0^2 + ||x[1:]||^2 = -1/c

        Args:
            x: (..., d+1) tensor where x[..., 0] is ignored

        Returns:
            x_hyp: (..., d+1) tensor on hyperboloid
        """
        x_space = x[..., 1:]
        space_sqnorm = (x_space ** 2).sum(dim=-1, keepdim=True)
        x_time = torch.sqrt(1.0 / self.c + space_sqnorm + self.eps)
        return torch.cat([x_time, x_space], dim=-1)

    def expmap0(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from origin's tangent space to hyperboloid.

        exp_o(v) = cosh(sqrt(c) * ||v||) * o + sinh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v

        where o = [1/sqrt(c), 0, ..., 0] is the origin on the hyperboloid.

        Args:
            v: (..., d+1) tangent vector at origin (v[..., 0] should be 0)

        Returns:
            x: (..., d+1) point on hyperboloid
        """
        v_space = v[..., 1:]
        v_norm = torch.clamp(torch.norm(v_space, dim=-1, keepdim=True), min=self.eps)

        # Scaled norm
        scaled_norm = self.sqrt_c * v_norm

        # Hyperbolic trig functions
        cosh_norm = torch.cosh(scaled_norm)
        sinh_norm = torch.sinh(scaled_norm)

        # Compute point on hyperboloid
        x_time = cosh_norm / self.sqrt_c
        x_space = sinh_norm / (scaled_norm + self.eps) * v_space

        return torch.cat([x_time, x_space], dim=-1)

    def logmap0(self, x: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from hyperboloid to origin's tangent space.

        log_o(x) = arccosh(sqrt(c) * x_0) / (sqrt(c) * sqrt(x_0^2 - 1/c)) * [0, x_1, ..., x_d]

        Args:
            x: (..., d+1) point on hyperboloid

        Returns:
            v: (..., d+1) tangent vector at origin
        """
        x_time = x[..., 0:1]
        x_space = x[..., 1:]

        # arccosh argument: sqrt(c) * x_0 >= 1
        arg = torch.clamp(self.sqrt_c * x_time, min=1.0 + self.eps)
        theta = torch.acosh(arg)

        # Denominator: sqrt(c) * sqrt(x_0^2 - 1/c) = sqrt(c * x_0^2 - 1)
        denom = torch.sqrt(torch.clamp(self.c * x_time ** 2 - 1, min=self.eps))

        # Scale factor
        scale = theta / (self.sqrt_c * denom + self.eps)

        v_space = scale * x_space
        v_time = torch.zeros_like(x_time)

        return torch.cat([v_time, v_space], dim=-1)

    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance between two points.

        d(x, y) = (1/sqrt(c)) * arccosh(-c * <x, y>_L)

        Args:
            x, y: (..., d+1) points on hyperboloid

        Returns:
            dist: (..., 1) hyperbolic distances
        """
        inner = self.minkowski_inner(x, y)
        # -c * <x,y>_L >= 1 for points on hyperboloid
        arg = torch.clamp(-self.c * inner, min=1.0 + self.eps)
        return torch.acosh(arg) / self.sqrt_c

    def parallel_transport0(self, v: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Parallel transport vector v from origin to point x.

        Args:
            v: (..., d+1) tangent vector at origin
            x: (..., d+1) destination point on hyperboloid

        Returns:
            v_x: (..., d+1) transported vector at x
        """
        # Origin on hyperboloid
        o_time = torch.ones_like(x[..., 0:1]) / self.sqrt_c
        o = torch.cat([o_time, torch.zeros_like(x[..., 1:])], dim=-1)

        # Lorentz factor
        alpha = -self.c * self.minkowski_inner(x, o)

        # Transport formula
        coef = self.c * self.minkowski_inner(x, v) / (alpha + 1)
        v_x = v + coef * (x + o)

        return v_x

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mobius addition in Lorentz model.

        Adds y to x via parallel transport: exp_x(PT_{o->x}(log_o(y)))

        Args:
            x, y: (..., d+1) points on hyperboloid

        Returns:
            z: (..., d+1) result on hyperboloid
        """
        v = self.logmap0(y)  # y in tangent space at origin
        v_x = self.parallel_transport0(v, x)  # transport to x

        # Exponential map at x (simplified using tangent vector norm)
        v_norm = self.minkowski_norm(v_x)
        scaled = self.sqrt_c * v_norm

        cosh_s = torch.cosh(scaled)
        sinh_s = torch.sinh(scaled)

        # exp_x(v) = cosh(||v||) * x + sinh(||v||) / ||v|| * v
        z = cosh_s * x + (sinh_s / (scaled + self.eps)) * v_x

        return z


class HyperbolicAttention(nn.Module):
    """Cross-attention using hyperbolic distance for attention weights.

    Projects features to Lorentz hyperboloid, computes attention weights
    based on hyperbolic distances (closer = higher attention), and
    aggregates in tangent space.

    Args:
        hidden_dim: Feature dimension
        num_heads: Number of attention heads
        curvature: Hyperbolic curvature (default: 1.0)
        dropout: Attention dropout probability
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        curvature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.curvature = curvature

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.manifold = LorentzManifold(curvature)

        # Projections (Euclidean)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Temperature for attention
        self.scale = nn.Parameter(torch.ones(num_heads) * math.sqrt(self.head_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            query: (L_q, D) query features (CDR)
            key: (L_k, D) key features (antigen)
            value: (L_k, D) value features (antigen)
            key_padding_mask: (L_k,) mask for padding (True = ignore)

        Returns:
            out: (L_q, D) attended features
        """
        L_q = query.shape[0]
        L_k = key.shape[0]

        # Project to Q, K, V
        q = self.q_proj(query)  # (L_q, D)
        k = self.k_proj(key)    # (L_k, D)
        v = self.v_proj(value)  # (L_k, D)

        # Reshape for multi-head: (L, H, head_dim)
        q = q.view(L_q, self.num_heads, self.head_dim)
        k = k.view(L_k, self.num_heads, self.head_dim)
        v = v.view(L_k, self.num_heads, self.head_dim)

        # Add time dimension for hyperboloid (prepend zeros, will be computed)
        # q_ext: (L_q, H, head_dim + 1)
        q_ext = torch.cat([torch.zeros(L_q, self.num_heads, 1, device=q.device), q], dim=-1)
        k_ext = torch.cat([torch.zeros(L_k, self.num_heads, 1, device=k.device), k], dim=-1)

        # Project to hyperboloid
        q_hyp = self.manifold.project_to_hyperboloid(q_ext)  # (L_q, H, head_dim+1)
        k_hyp = self.manifold.project_to_hyperboloid(k_ext)  # (L_k, H, head_dim+1)

        # Compute pairwise hyperbolic distances
        # Expand for broadcasting: (L_q, 1, H, d+1) and (1, L_k, H, d+1)
        q_exp = q_hyp.unsqueeze(1)  # (L_q, 1, H, d+1)
        k_exp = k_hyp.unsqueeze(0)  # (1, L_k, H, d+1)

        # Hyperbolic distance: (L_q, L_k, H, 1)
        dists = self.manifold.hyperbolic_distance(q_exp, k_exp)
        dists = dists.squeeze(-1)  # (L_q, L_k, H)

        # Convert distance to attention: closer = higher attention
        # Use negative distance scaled by learnable temperature
        attn_logits = -dists / self.scale.view(1, 1, self.num_heads)  # (L_q, L_k, H)

        # Apply key padding mask if provided
        if key_padding_mask is not None:
            # key_padding_mask: (L_k,) -> (1, L_k, 1)
            mask = key_padding_mask.view(1, L_k, 1)
            attn_logits = attn_logits.masked_fill(mask, float('-inf'))

        # Softmax over keys
        attn = F.softmax(attn_logits, dim=1)  # (L_q, L_k, H)
        attn = self.dropout(attn)

        # Weighted aggregation in Euclidean space (tangent space approximation)
        # v: (L_k, H, head_dim), attn: (L_q, L_k, H)
        # out: (L_q, H, head_dim)
        out = torch.einsum('qkh,khd->qhd', attn, v)

        # Reshape and project output
        out = out.reshape(L_q, self.hidden_dim)
        out = self.out_proj(out)

        return out


class HyperbolicMessagePassing(nn.Module):
    """Message passing layer in hyperbolic space.

    Aggregates neighbor features using hyperbolic operations:
    1. Map neighbors to tangent space at each node
    2. Aggregate in tangent space (weighted by hyperbolic distance)
    3. Map back to hyperboloid

    Args:
        hidden_dim: Feature dimension
        curvature: Hyperbolic curvature
    """

    def __init__(self, hidden_dim: int, curvature: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.manifold = LorentzManifold(curvature)

        # Message MLP (operates in tangent space)
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for time component
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim + 1),
        )

        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * (hidden_dim + 1), hidden_dim + 1),
            nn.SiLU(),
            nn.Linear(hidden_dim + 1, hidden_dim + 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (N, D+1) node features on hyperboloid
            edge_index: (2, E) edge indices [source, target]
            edge_weight: (E,) optional edge weights

        Returns:
            x_out: (N, D+1) updated features on hyperboloid
        """
        row, col = edge_index
        N = x.shape[0]

        # Get source and target features
        x_src = x[row]  # (E, D+1)
        x_tgt = x[col]  # (E, D+1)

        # Map source to tangent space at target
        v_src = self.manifold.logmap0(x_src)

        # Compute hyperbolic distances for weighting
        dists = self.manifold.hyperbolic_distance(x_src, x_tgt).squeeze(-1)  # (E,)
        weights = F.softmax(-dists, dim=0)  # closer = higher weight

        if edge_weight is not None:
            weights = weights * edge_weight

        # Transform messages
        messages = self.message_mlp(v_src)  # (E, D+1)
        messages = messages * weights.unsqueeze(-1)

        # Aggregate messages per target node
        agg = torch.zeros(N, self.hidden_dim + 1, device=x.device)
        agg.scatter_add_(0, col.unsqueeze(-1).expand_as(messages), messages)

        # Map current node to tangent space
        v_self = self.manifold.logmap0(x)

        # Update: combine self and aggregated
        v_update = self.update_mlp(torch.cat([v_self, agg], dim=-1))

        # Map back to hyperboloid
        x_out = self.manifold.expmap0(v_update)

        return x_out
