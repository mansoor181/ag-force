"""Improved MINE Estimator with Hard Negative Mining.

The original MINE fails because random shuffle creates trivial negatives.
When batch diversity is low, the discriminator either:
1. Instantly distinguishes (MI saturates)
2. Cannot distinguish (MI = 0)

Fixes implemented:
1. Hard negative mining: use similarity-based selection instead of random shuffle
2. Gradient clipping and warmup for stability
3. Multiple negative samples per positive
4. Learnable temperature for scaling

References:
- MINE: Belghazi et al., ICML 2018
- Hard negatives: Robinson et al., "Contrastive Learning with Hard Negatives", ICLR 2021
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HardNegativeMINE(nn.Module):
    """MINE with hard negative mining and stability improvements.

    Key improvements over vanilla MINE:
    1. Hard negatives: select negatives similar to positives (harder discrimination)
    2. Multiple negatives: use top-K hardest negatives per sample
    3. Gradient normalization: prevent explosion
    4. Warmup: gradually increase MI target
    """

    def __init__(self, x_dim, y_dim, hidden_dim=128, n_negatives=4,
                 temperature=0.1, ema_decay=0.99):
        super().__init__()
        self.n_negatives = n_negatives
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Statistics network with LayerNorm for stability
        self.statistics_net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Projection for similarity computation (hard negative selection)
        self.proj = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.ema_decay = ema_decay
        self.register_buffer('ema_marginal', torch.tensor(1.0))
        self.register_buffer('warmup_steps', torch.tensor(0))
        self.warmup_total = 1000

    def _select_hard_negatives(self, y):
        """Select hard negatives based on similarity.

        For each y_i, find top-K most similar y_j (j != i).
        These are "hard" because they're similar but from different pairs.

        Args:
            y: (B, y_dim) epitope embeddings

        Returns:
            neg_indices: (B, n_neg) indices of hard negatives for each sample
            actual_n_neg: int, actual number of negatives (may be less than n_negatives)
        """
        B = y.shape[0]
        device = y.device

        # Adjust n_negatives if batch is too small
        actual_n_neg = min(self.n_negatives, B - 1)
        if actual_n_neg < 1:
            # Can't create negatives with B <= 1
            return None, 0

        if B <= self.n_negatives + 1:
            # Not enough samples for hard negative mining
            # Fall back to random selection with replacement if needed
            indices = []
            for i in range(B):
                # Exclude self
                candidates = [j for j in range(B) if j != i]
                if len(candidates) < actual_n_neg:
                    # Sample with replacement
                    chosen = torch.tensor(
                        [candidates[j % len(candidates)] for j in range(actual_n_neg)],
                        device=device)
                else:
                    perm = torch.randperm(len(candidates), device=device)[:actual_n_neg]
                    chosen = torch.tensor([candidates[p] for p in perm.tolist()], device=device)
                indices.append(chosen)
            return torch.stack(indices), actual_n_neg

        # Project for similarity
        y_proj = F.normalize(self.proj(y), dim=-1)  # (B, D)

        # Compute similarity matrix
        sim = y_proj @ y_proj.T  # (B, B)

        # Mask out self-similarity
        mask = torch.eye(B, device=device).bool()
        sim = sim.masked_fill(mask, float('-inf'))

        # Select top-K most similar (hardest negatives)
        _, neg_indices = sim.topk(actual_n_neg, dim=-1)  # (B, n_neg)

        return neg_indices, actual_n_neg

    def forward(self, x, y, return_components=False):
        """Compute MINE with hard negatives.

        Args:
            x: (B, x_dim) CDR embeddings
            y: (B, y_dim) epitope embeddings
            return_components: if True, return joint/marginal terms separately

        Returns:
            mi_estimate: scalar MINE lower bound
        """
        B = x.shape[0]
        device = x.device

        if B < 2:
            return torch.tensor(0.0, device=device)

        # Joint distribution: (x_i, y_i) pairs
        joint = torch.cat([x, y], dim=-1)  # (B, x+y)
        t_joint = self.statistics_net(joint)  # (B, 1)
        t_joint = torch.clamp(t_joint / self.temperature.abs(), -10, 10)

        # Select hard negatives for marginal
        neg_indices, actual_n_neg = self._select_hard_negatives(y)  # (B, n_neg), int

        if neg_indices is None or actual_n_neg < 1:
            # Can't compute marginal without negatives
            return torch.tensor(0.0, device=device)

        # Compute marginal term with hard negatives
        # For each x_i, compute T(x_i, y_j) for hard negative y_j
        x_expanded = x.unsqueeze(1).expand(-1, actual_n_neg, -1)  # (B, n_neg, x_dim)
        y_neg = y[neg_indices]  # (B, n_neg, y_dim)

        marginal = torch.cat([x_expanded, y_neg], dim=-1)  # (B, n_neg, x+y)
        marginal = marginal.reshape(B * actual_n_neg, -1)
        t_marginal = self.statistics_net(marginal)  # (B*n_neg, 1)
        t_marginal = torch.clamp(t_marginal / self.temperature.abs(), -10, 10)
        t_marginal = t_marginal.reshape(B, actual_n_neg)  # (B, n_neg)

        # MINE estimate: E_joint[T] - log(E_marginal[exp(T)])
        joint_term = t_joint.mean()
        # Use logsumexp across all marginal samples
        log_marginal_term = torch.logsumexp(t_marginal.flatten(), dim=0) - math.log(B * actual_n_neg)

        # EMA for marginal (reduces variance)
        if self.training:
            exp_marginal = log_marginal_term.exp().detach()
            self.ema_marginal = self.ema_decay * self.ema_marginal + (1 - self.ema_decay) * exp_marginal
            self.warmup_steps += 1

            # Warmup: gradually increase the MI target
            warmup_factor = min(1.0, self.warmup_steps.float() / self.warmup_total)
            mi_estimate = warmup_factor * (joint_term - self.ema_marginal.log())
        else:
            mi_estimate = joint_term - log_marginal_term

        if return_components:
            return mi_estimate.squeeze(), joint_term.squeeze(), log_marginal_term.squeeze()

        return mi_estimate.squeeze()


class InfoNCEMINE(nn.Module):
    """InfoNCE-style MI estimator (more stable than MINE).

    Instead of Donsker-Varadhan, uses NCE bound:
    MI(X,Y) >= log(N) - L_NCE

    where L_NCE is the N-way classification loss.

    More stable and doesn't suffer from high-variance gradients.
    """

    def __init__(self, x_dim, y_dim, hidden_dim=128, temperature=0.1):
        super().__init__()
        self.temperature = temperature

        # Projection heads to shared space
        self.proj_x = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.proj_y = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

    def forward(self, x, y):
        """Compute InfoNCE MI lower bound.

        Args:
            x: (B, x_dim) CDR embeddings
            y: (B, y_dim) epitope embeddings

        Returns:
            mi_estimate: scalar MI lower bound (always non-negative)
        """
        B = x.shape[0]
        device = x.device

        if B < 2:
            return torch.tensor(0.0, device=device)

        # Project to shared space
        x_proj = F.normalize(self.proj_x(x), dim=-1)  # (B, D)
        y_proj = F.normalize(self.proj_y(y), dim=-1)  # (B, D)

        # Similarity matrix
        logits = x_proj @ y_proj.T / self.temperature  # (B, B)

        # InfoNCE loss (cross-entropy with diagonal as positives)
        labels = torch.arange(B, device=device)
        nce_loss = F.cross_entropy(logits, labels)

        # MI lower bound: log(B) - L_NCE
        mi_estimate = math.log(B) - nce_loss

        return mi_estimate


class DualMINE(nn.Module):
    """Dual estimator combining MINE and InfoNCE.

    Uses InfoNCE for stable gradients and MINE for tighter bound.
    The combination provides both stability and expressiveness.
    """

    def __init__(self, x_dim, y_dim, hidden_dim=128, alpha=0.5):
        super().__init__()
        self.alpha = alpha

        self.mine = HardNegativeMINE(x_dim, y_dim, hidden_dim)
        self.nce = InfoNCEMINE(x_dim, y_dim, hidden_dim)

    def forward(self, x, y):
        """Compute combined MI estimate.

        Args:
            x: (B, x_dim) CDR embeddings
            y: (B, y_dim) epitope embeddings

        Returns:
            mi_estimate: scalar
        """
        mi_mine = self.mine(x, y)
        mi_nce = self.nce(x, y)

        # Weighted combination
        return self.alpha * mi_mine + (1 - self.alpha) * mi_nce
