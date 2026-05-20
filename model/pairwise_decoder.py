"""Pairwise Sequence Decoder with Iterative Refinement.

Addresses the fundamental limitation of independent per-position CE loss:
1. PairwiseSequenceHead: Potts-like model with learned pairwise couplings J_ij
2. IterativeRefinementDecoder: Multi-pass decoding with Gumbel-softmax feedback

Key insight: Contact positions have correlated AA preferences.
  - Position i prefers Arg only if neighbor j is NOT Arg (charge clash)
  - Pairwise scoring: E = Σ h_i(AA_i) + Σ J_ij(AA_i, AA_j)
  - Iterative refinement: breaks conditional independence

References:
  - Potts models for protein: Marks et al., "Protein 3D structure from MSA", PNAS 2011
  - Gumbel-softmax: Jang et al., "Categorical Reparameterization", ICLR 2017
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseSequenceHead(nn.Module):
    """Potts-like sequence head with pairwise AA couplings.

    Energy function: E(seq) = Σ_i h_i(AA_i) + Σ_{i<j} J_ij(AA_i, AA_j)

    Implementation:
    - h_i: (L, 20) single-site potentials from MLP
    - J_ij: learned 20x20 coupling matrices for adjacent positions
    - Message passing: refine h_i using J_ij couplings

    Two coupling modes:
    1. 'adjacent': only i, i+1 pairs (linear complexity)
    2. 'full': all pairs within window (quadratic but bounded)
    """

    def __init__(self, hidden_dim, n_aa=25, coupling_mode='adjacent',
                 n_message_passes=2, window_size=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_aa = n_aa
        self.coupling_mode = coupling_mode
        self.n_message_passes = n_message_passes
        self.window_size = window_size

        # Single-site potential: hidden -> AA logits
        self.h_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_aa),
        )

        # Pairwise coupling matrix (20x20 for each relative position)
        # J[d] gives couplings for positions i and i+d
        if coupling_mode == 'adjacent':
            # Only i, i+1 pairs
            self.J = nn.Parameter(torch.zeros(1, n_aa, n_aa))
            nn.init.normal_(self.J, std=0.01)
        else:
            # Pairs within window
            self.J = nn.Parameter(torch.zeros(window_size, n_aa, n_aa))
            nn.init.normal_(self.J, std=0.01)

        # Optional: position-dependent coupling strength
        self.coupling_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, h, mask=None):
        """Compute pairwise-refined logits.

        Args:
            h: (L, D) hidden features
            mask: (L,) optional mask for valid positions

        Returns:
            logits: (L, n_aa) refined AA logits
        """
        L = h.shape[0]

        # Initial single-site potentials
        h_single = self.h_proj(h)  # (L, n_aa)

        # Message passing to incorporate pairwise couplings
        logits = h_single
        for _ in range(self.n_message_passes):
            logits = self._message_pass(logits, h)

        return logits

    def _message_pass(self, logits, h):
        """One round of message passing with pairwise couplings.

        Update: h'_i = h_i + Σ_j J_ij @ softmax(h_j)
        """
        L, n_aa = logits.shape

        # Current beliefs (soft assignments)
        beliefs = F.softmax(logits, dim=-1)  # (L, n_aa)

        # Messages from neighbors
        messages = torch.zeros_like(logits)

        if self.coupling_mode == 'adjacent':
            # Only consider i, i+1 pairs
            J = self.J[0]  # (n_aa, n_aa)

            # Forward messages: j -> i where j = i+1
            if L > 1:
                # Gate coupling strength based on hidden features
                h_pairs = torch.cat([h[:-1], h[1:]], dim=-1)  # (L-1, 2D)
                gates = self.coupling_gate(h_pairs)  # (L-1, 1)

                # Message from j=i+1 to i
                msg_forward = beliefs[1:] @ J.T  # (L-1, n_aa)
                messages[:-1] += gates * msg_forward

                # Message from j=i-1 to i
                msg_backward = beliefs[:-1] @ J  # (L-1, n_aa)
                messages[1:] += gates * msg_backward
        else:
            # Full window coupling
            for d in range(1, min(self.window_size, L)):
                J_d = self.J[d-1]  # (n_aa, n_aa)

                # Forward: j=i+d to i
                if d < L:
                    msg = beliefs[d:] @ J_d.T
                    messages[:-d] += msg

                # Backward: j=i-d to i
                if d < L:
                    msg = beliefs[:-d] @ J_d
                    messages[d:] += msg

        # Update logits
        return logits + messages

    def pairwise_loss(self, logits, targets, lambda_pair=0.1):
        """Compute loss with pairwise consistency term.

        L = CE(logits, targets) + λ * inconsistency_penalty

        The inconsistency penalty encourages adjacent predictions
        to have low pairwise energy (be compatible).
        """
        # Standard CE loss
        ce_loss = F.cross_entropy(logits, targets, reduction='mean')

        # Pairwise consistency: penalize high-energy adjacent pairs
        L = logits.shape[0]
        if L < 2:
            return ce_loss

        probs = F.softmax(logits, dim=-1)  # (L, n_aa)

        if self.coupling_mode == 'adjacent':
            J = self.J[0]  # (n_aa, n_aa)
            # Energy for each adjacent pair: p_i^T @ J @ p_{i+1}
            # Lower is better (more compatible)
            pair_energy = torch.einsum('ia,ab,ib->i', probs[:-1], J, probs[1:])
            # Penalize high energy (incompatible pairs)
            pair_penalty = pair_energy.mean()
        else:
            pair_penalty = 0
            for d in range(1, min(self.window_size, L)):
                J_d = self.J[d-1]
                if d < L:
                    energy = torch.einsum('ia,ab,ib->i', probs[:-d], J_d, probs[d:])
                    pair_penalty += energy.mean()
            pair_penalty /= min(self.window_size, L) - 1

        return ce_loss + lambda_pair * pair_penalty


class IterativeRefinementDecoder(nn.Module):
    """Multi-pass iterative refinement decoder.

    Breaks conditional independence by letting each position see
    predictions from previous pass:

    Pass 1: P(AA_i | antigen)
    Pass 2: P(AA_i | antigen, seq_pass1)
    Pass 3: P(AA_i | antigen, seq_pass2)

    Uses Gumbel-softmax for differentiable sampling during training.
    """

    def __init__(self, hidden_dim, n_aa=25, n_passes=3,
                 gumbel_tau_init=1.0, gumbel_tau_min=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_aa = n_aa
        self.n_passes = n_passes
        self.gumbel_tau = gumbel_tau_init
        self.gumbel_tau_min = gumbel_tau_min

        # AA embedding for feedback from previous pass
        self.aa_embed = nn.Embedding(n_aa, hidden_dim // 2)

        # Soft AA embedding for Gumbel-softmax outputs
        self.soft_aa_proj = nn.Linear(n_aa, hidden_dim // 2)

        # Refinement layers (one per pass after first)
        self.refinement_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_passes - 1)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_aa),
        )

        # Pass embedding (helps model know which refinement pass)
        self.pass_embed = nn.Embedding(n_passes, hidden_dim)

    def anneal_temperature(self, factor=0.95, min_tau=None):
        """Anneal Gumbel temperature during training."""
        min_val = min_tau if min_tau is not None else self.gumbel_tau_min
        self.gumbel_tau = max(min_val, self.gumbel_tau * factor)

    def forward(self, h, targets=None, return_all_passes=False):
        """Multi-pass refinement.

        Args:
            h: (L, D) hidden features from encoder
            targets: (L,) ground truth for loss (training only)
            return_all_passes: if True, return logits from all passes

        Returns:
            final_logits: (L, n_aa)
            loss: scalar (if targets provided)
            all_logits: list of (L, n_aa) for each pass (if return_all_passes)
        """
        L = h.shape[0]
        device = h.device

        all_logits = []
        all_losses = []

        # Current hidden state
        h_current = h
        prev_soft = None

        for pass_idx in range(self.n_passes):
            # Add pass embedding
            pass_emb = self.pass_embed(torch.tensor([pass_idx], device=device))
            h_pass = h_current + pass_emb.expand(L, -1)

            if pass_idx > 0 and prev_soft is not None:
                # Incorporate feedback from previous pass
                soft_emb = self.soft_aa_proj(prev_soft)  # (L, D/2)
                h_combined = torch.cat([h_pass, soft_emb], dim=-1)
                h_pass = self.refinement_layers[pass_idx - 1](h_combined)

            # Compute logits for this pass
            logits = self.output_head(h_pass)  # (L, n_aa)
            all_logits.append(logits)

            # Compute loss if targets provided
            if targets is not None:
                loss = F.cross_entropy(logits, targets, reduction='mean')
                # Weight later passes more (they should be better)
                weight = 1.0 + 0.5 * pass_idx
                all_losses.append(weight * loss)

            # Prepare feedback for next pass
            if pass_idx < self.n_passes - 1:
                if self.training:
                    # Gumbel-softmax for differentiable sampling
                    prev_soft = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False)
                else:
                    # Hard selection during inference
                    prev_soft = F.one_hot(logits.argmax(dim=-1), self.n_aa).float()

        final_logits = all_logits[-1]

        if targets is not None:
            # Total loss: weighted sum across passes
            total_loss = sum(all_losses) / len(all_losses)
            if return_all_passes:
                return final_logits, total_loss, all_logits
            return final_logits, total_loss

        if return_all_passes:
            return final_logits, all_logits
        return final_logits

    def generate(self, h, greedy=True):
        """Generate sequence with iterative refinement.

        Args:
            h: (L, D) hidden features
            greedy: if True, use argmax; else sample

        Returns:
            seq: (L,) predicted AA indices
            logits: (L, n_aa) final logits
        """
        with torch.no_grad():
            final_logits = self.forward(h)

        if greedy:
            seq = final_logits.argmax(dim=-1)
        else:
            probs = F.softmax(final_logits, dim=-1)
            seq = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return seq, final_logits


class CombinedPairwiseIterativeHead(nn.Module):
    """Combined pairwise + iterative refinement sequence head.

    Architecture:
    1. Initial logits from MLP
    2. Iterative refinement (n_passes)
    3. Pairwise message passing at each pass

    This captures both:
    - Long-range dependencies via iterative refinement
    - Local chemical constraints via pairwise couplings
    """

    def __init__(self, hidden_dim, n_aa=25, n_passes=3, n_message_passes=2,
                 coupling_mode='adjacent', window_size=5,
                 gumbel_tau_init=1.0, lambda_pair=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_aa = n_aa
        self.n_passes = n_passes
        self.lambda_pair = lambda_pair
        self.gumbel_tau = gumbel_tau_init

        # Pairwise head for each refinement pass
        self.pairwise_heads = nn.ModuleList([
            PairwiseSequenceHead(
                hidden_dim, n_aa,
                coupling_mode=coupling_mode,
                n_message_passes=n_message_passes,
                window_size=window_size
            )
            for _ in range(n_passes)
        ])

        # Feedback embedding
        self.soft_aa_proj = nn.Linear(n_aa, hidden_dim // 2)

        # Refinement layers
        self.refinement_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            )
            for _ in range(n_passes - 1)
        ])

        # Pass embedding
        self.pass_embed = nn.Embedding(n_passes, hidden_dim)

    def forward(self, h, targets=None):
        """Forward with combined pairwise + iterative refinement.

        Args:
            h: (L, D) hidden features
            targets: (L,) ground truth (for loss)

        Returns:
            logits: (L, n_aa)
            loss: scalar (if targets provided)
        """
        L = h.shape[0]
        device = h.device

        all_losses = []
        h_current = h
        prev_soft = None

        for pass_idx in range(self.n_passes):
            # Add pass embedding
            pass_emb = self.pass_embed(torch.tensor([pass_idx], device=device))
            h_pass = h_current + pass_emb.expand(L, -1)

            # Incorporate feedback from previous pass
            if pass_idx > 0 and prev_soft is not None:
                soft_emb = self.soft_aa_proj(prev_soft)
                h_combined = torch.cat([h_pass, soft_emb], dim=-1)
                h_pass = self.refinement_layers[pass_idx - 1](h_combined)

            # Pairwise-refined logits
            logits = self.pairwise_heads[pass_idx](h_pass)

            # Loss with pairwise consistency
            if targets is not None:
                loss = self.pairwise_heads[pass_idx].pairwise_loss(
                    logits, targets, lambda_pair=self.lambda_pair)
                weight = 1.0 + 0.5 * pass_idx
                all_losses.append(weight * loss)

            # Prepare feedback
            if pass_idx < self.n_passes - 1:
                if self.training:
                    prev_soft = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False)
                else:
                    prev_soft = F.one_hot(logits.argmax(dim=-1), self.n_aa).float()

        if targets is not None:
            total_loss = sum(all_losses) / len(all_losses)
            return logits, total_loss

        return logits

    def anneal_temperature(self, factor=0.95, min_tau=0.1):
        """Anneal Gumbel temperature."""
        self.gumbel_tau = max(min_tau, self.gumbel_tau * factor)


class MDNPairwiseHead(nn.Module):
    """MDN + AMCL + Pairwise scoring sequence head.

    Combines:
    1. MDN: K mixture components to capture multiple modes
    2. AMCL: Annealed soft-to-hard winner-take-all training
    3. Pairwise: Potts-like J_ij couplings for local AA constraints

    Each component k has:
    - h_proj_k: hidden -> logits
    - J_k: (n_aa, n_aa) symmetric coupling matrix
    - Message passing refines logits using neighbor beliefs

    Loss options:
    - MDN: -log(sum_k pi_k * P(target|k))
    - AMCL: sum_k w_k * (CE_k + lambda * pairwise_energy_k)
    """

    def __init__(self, hidden_dim, num_aa, K=4, dropout=0.1,
                 n_message_passes=2, lambda_pair=0.1, vocab=None):
        super().__init__()
        self.K = K
        self.num_aa = num_aa
        self.n_message_passes = n_message_passes
        self.lambda_pair = lambda_pair

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        # K component heads (each produces initial logits)
        self.component_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, num_aa) for _ in range(K)
        ])

        # Mixing weights head
        self.pi_head = nn.Linear(hidden_dim // 2, K)

        # Pairwise coupling matrices (one per component, symmetric)
        # J_k[a,b] = coupling energy for AA types a and b at adjacent positions
        # Initialized near zero, will learn compatible/incompatible pairs
        self.J = nn.ParameterList([
            nn.Parameter(torch.zeros(num_aa, num_aa) + 0.01 * torch.randn(num_aa, num_aa))
            for _ in range(K)
        ])

        # Position-aware gating for coupling strength (input is hidden_dim//2 after shared)
        self.coupling_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Mask special tokens
        if vocab is not None:
            special_mask = torch.tensor(vocab.get_special_mask(), dtype=torch.bool)
            self.register_buffer('special_token_mask', special_mask)

    def _symmetrize_J(self, J):
        """Ensure J is symmetric: J_sym = (J + J^T) / 2."""
        return (J + J.t()) / 2

    def _message_pass(self, logits, h, J):
        """One round of belief propagation with pairwise couplings.

        Update: logits'_i = logits_i + sum_j gate_ij * J @ softmax(logits_j)

        Args:
            logits: (L, n_aa) current logits
            h: (L, D/2) hidden features for gating
            J: (n_aa, n_aa) symmetric coupling matrix
        """
        L = logits.shape[0]
        if L < 2:
            return logits

        J_sym = self._symmetrize_J(J)
        beliefs = F.softmax(logits, dim=-1)  # (L, n_aa)

        # Compute gates based on hidden features
        gates = self.coupling_gate(h)  # (L, 1)

        # Messages from neighbors (bidirectional)
        messages = torch.zeros_like(logits)

        # Forward: message from i+1 to i
        msg_forward = beliefs[1:] @ J_sym  # (L-1, n_aa)
        messages[:-1] += gates[:-1] * msg_forward

        # Backward: message from i-1 to i
        msg_backward = beliefs[:-1] @ J_sym  # (L-1, n_aa)
        messages[1:] += gates[1:] * msg_backward

        return logits + messages

    def _compute_pairwise_energy(self, logits, J):
        """Compute pairwise energy for adjacent positions.

        E = sum_{i} p_i^T @ J @ p_{i+1}

        Lower energy = more compatible (J should learn negative values for good pairs).
        """
        L = logits.shape[0]
        if L < 2:
            return torch.tensor(0.0, device=logits.device)

        J_sym = self._symmetrize_J(J)
        probs = F.softmax(logits, dim=-1)

        # Energy for each adjacent pair
        energy = torch.einsum('ia,ab,ib->i', probs[:-1], J_sym, probs[1:])
        return energy.mean()

    def forward(self, x, hard_mask=True):
        """Compute mixing weights and pairwise-refined logits for each component.

        Args:
            x: (L, hidden_dim) input features
            hard_mask: if True, use -inf for special tokens

        Returns:
            pi: (L, K) mixing weights
            logits_list: list of K tensors, each (L, num_aa)
        """
        h = self.shared(x)  # (L, D/2)
        pi = F.softmax(self.pi_head(h), dim=-1)

        logits_list = []
        for k in range(self.K):
            # Initial logits from component head
            logits_k = self.component_heads[k](h)

            # Refine with message passing using component's J matrix
            for _ in range(self.n_message_passes):
                logits_k = self._message_pass(logits_k, h, self.J[k])

            logits_list.append(logits_k)

        # Mask special tokens
        if hasattr(self, 'special_token_mask'):
            penalty = float('-inf') if hard_mask else -10.0
            logits_list = [logits.masked_fill(self.special_token_mask, penalty)
                           for logits in logits_list]

        return pi, logits_list

    def mdn_loss(self, x, target):
        """Standard MDN loss: -log(sum_k pi_k * P(target|k))."""
        pi, logits_list = self.forward(x, hard_mask=False)

        logits = torch.stack(logits_list, dim=0)  # (K, L, num_aa)
        log_probs = F.log_softmax(logits, dim=-1)

        target_expanded = target.unsqueeze(0).expand(self.K, -1)
        log_prob_target = log_probs.gather(dim=-1, index=target_expanded.unsqueeze(-1)).squeeze(-1)

        pi_t = pi.t()  # (K, L)
        log_pi = torch.log(pi_t + 1e-10)
        log_mixture = log_pi + log_prob_target
        log_likelihood = torch.logsumexp(log_mixture, dim=0)

        return -log_likelihood.mean()

    def amcl_loss(self, x, target, tau):
        """AMCL loss with pairwise consistency.

        Loss = sum_k w_k * (CE_k + lambda * pairwise_energy_k)
        where w_k = softmax(-loss_k / tau)
        """
        h = self.shared(x)
        logits_list = []

        for k in range(self.K):
            logits_k = self.component_heads[k](h)
            for _ in range(self.n_message_passes):
                logits_k = self._message_pass(logits_k, h, self.J[k])
            logits_list.append(logits_k)

        # Mask special tokens
        if hasattr(self, 'special_token_mask'):
            logits_list = [logits.masked_fill(self.special_token_mask, -10.0)
                           for logits in logits_list]

        # Compute per-component loss (CE + pairwise)
        per_component_loss = []
        for k, logits_k in enumerate(logits_list):
            ce_k = F.cross_entropy(logits_k, target, reduction='mean')

            # Pairwise energy: we want to MINIMIZE energy for compatible pairs
            # So we add positive energy as penalty (incompatible pairs have high energy)
            pair_energy_k = self._compute_pairwise_energy(logits_k, self.J[k])

            loss_k = ce_k + self.lambda_pair * pair_energy_k
            per_component_loss.append(loss_k)

        losses = torch.stack(per_component_loss)

        # Boltzmann soft assignment (detached)
        with torch.no_grad():
            weights = F.softmax(-losses / tau, dim=0)

        # Weighted loss
        return (weights * losses).sum()

    def sample(self, x, greedy=True):
        """Sample from the MDN."""
        pi, logits_list = self.forward(x, hard_mask=True)
        L = x.shape[0]
        logits = torch.stack(logits_list, dim=0)

        if greedy:
            best_k = pi.argmax(dim=-1)
            logits_selected = logits[best_k, torch.arange(L, device=x.device), :]
            samples = logits_selected.argmax(dim=-1)
        else:
            k_samples = torch.multinomial(pi, num_samples=1).squeeze(-1)
            logits_selected = logits[k_samples, torch.arange(L, device=x.device), :]
            probs = F.softmax(logits_selected, dim=-1)
            samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return samples

    def get_logits(self, x):
        """Get combined logits (weighted average of components)."""
        pi, logits_list = self.forward(x, hard_mask=True)
        logits = torch.stack(logits_list, dim=0)  # (K, L, num_aa)
        pi_expanded = pi.t().unsqueeze(-1)  # (K, L, 1)
        return (pi_expanded * logits).sum(dim=0)
