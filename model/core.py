"""v32: Cycle-Consistent Antigen Reconstruction + Direct Attention.

Paper 4 -- Failure Mode RC2 (Antigen Ignorance)

v32 CHANGES (based on empirical diagnosis):
  1. REMOVE AntigenGatedBottleneck -- gate contributes 0 pp AAR (ablation proven)
  2. Seq_head reads ONLY attn_out -- force all signal through cross-attention
  3. ADD AntigenClassificationHead -- cycle consistency: pred_S -> antigen_id
  4. DISABLE MINE/InfoNCE/latent coupling -- empirically inert (optimized but no AAR transfer)
  5. Slower LR decay (anneal_base 0.97) -- prevent premature convergence

Architecture:
    AA_embedding(S) -> VirtualNodeEGNN(H_0, X, edges, masks)
    -> framework_dropout(aa_embd) [training only]
    -> cross_attn(CDR_query, AG_key/value) -> attn_out
    -> seq_head(attn_out) -> logits   [NO gated_cdr bypass]
    -> antigen_cls_head(pred_probs) -> antigen_id loss  [CYCLE CONSISTENCY]

Key v32 insight: The gated path contributed ZERO to AAR (gate=0 ablation unchanged).
By removing the bypass, gradients MUST flow through cross-attention, forcing antigen use.

The antigen classification head forces: if model can't tell which antigen given its CDR,
it isn't actually conditioning on antigen. This directly addresses vocab collapse.

Losses (13-tuple): seq, coord, pairing, dock, shadow, infonce, mine, gdpp,
                    latent_mse, latent_seq, aux, antigen_cls
"""
import math
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_sum
import numpy as np

# Path setup for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
_GENBIO_ROOT = os.path.normpath(os.path.join(_CODE_DIR, '..', '..'))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
if _GENBIO_ROOT not in sys.path:
    sys.path.insert(0, _GENBIO_ROOT)

from data.pdb_utils import VOCAB
from model.modules import VirtualNodeEGNN
from model.kan_layers import KANLinear
from model.hyperbolic import HyperbolicAttention
from model.topo_features import TopologicalBindingSignature, GUDHI_AVAILABLE
from model.pairwise_decoder import CombinedPairwiseIterativeHead, PairwiseSequenceHead, IterativeRefinementDecoder, MDNPairwiseHead
from model.improved_mine import HardNegativeMINE, InfoNCEMINE, DualMINE
from shared.sparse_features import ProteinFeatureEncoderWithSparse, CDRFeaturePredictor


# ══════════════════════════════════════════════════════════════════════════════
# BLOSUM62 Substitution Matrix for MDN weighted loss
# ══════════════════════════════════════════════════════════════════════════════

BLOSUM62_RAW = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
    'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
    'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
    'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4},
}

AA_ORDER_STANDARD = 'ACDEFGHIKLMNPQRSTVWY'


def get_blosum_weight_matrix(vocab=None, vocab_size=None, device='cpu', min_weight=0.2, max_weight=1.0):
    """Create BLOSUM-based weight matrix for loss computation."""
    n_aa = 20
    blosum_np = np.zeros((n_aa, n_aa))
    for i, aa_i in enumerate(AA_ORDER_STANDARD):
        for j, aa_j in enumerate(AA_ORDER_STANDARD):
            blosum_np[i, j] = BLOSUM62_RAW[aa_i][aa_j]

    blosum_min = blosum_np.min()
    blosum_max = blosum_np.max()
    weight_np = max_weight - (blosum_np - blosum_min) / (blosum_max - blosum_min) * (max_weight - min_weight)

    if vocab_size is None:
        vocab_size = len(vocab) if vocab is not None else n_aa

    weight_matrix = torch.full((vocab_size, vocab_size), max_weight, dtype=torch.float32, device=device)

    if vocab is not None:
        for i, aa_i in enumerate(AA_ORDER_STANDARD):
            vocab_i = vocab.symbol_to_idx(aa_i)
            if vocab_i >= vocab_size:
                continue
            for j, aa_j in enumerate(AA_ORDER_STANDARD):
                vocab_j = vocab.symbol_to_idx(aa_j)
                if vocab_j >= vocab_size:
                    continue
                weight_matrix[vocab_i, vocab_j] = weight_np[i, j]
    else:
        weight_matrix[:n_aa, :n_aa] = torch.tensor(weight_np, dtype=torch.float32, device=device)

    return weight_matrix


# ══════════════════════════════════════════════════════════════════════════════
# MDN Sequence Head (addresses mode collapse)
# ══════════════════════════════════════════════════════════════════════════════

class MDNSeqHead(nn.Module):
    """Mixture Density Network sequence head.

    Outputs K mixture components, each producing categorical logits over amino acids.
    The mixing weights pi determine how to combine the K components.

    Loss: -log(sum_k pi_k * P(target | component k))
    """
    def __init__(self, hidden_dim, num_aa, K=4, dropout=0.1, vocab=None):
        super().__init__()
        self.K = K
        self.num_aa = num_aa

        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        # K component heads (each produces logits over num_aa)
        self.component_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, num_aa) for _ in range(K)
        ])
        # Mixing weights head (produces K weights per position)
        self.pi_head = nn.Linear(hidden_dim // 2, K)

        # Mask special tokens (indices 0-5 in VOCAB: PAD, UNK, BOA, BOH, BOL, SEP)
        if vocab is not None:
            special_mask = torch.tensor(vocab.get_special_mask(), dtype=torch.bool)
            self.register_buffer('special_token_mask', special_mask)

    def forward(self, x, hard_mask=True):
        """Compute mixing weights and component logits.

        Args:
            x: (L, hidden_dim) input features
            hard_mask: if True, use -inf for special tokens

        Returns:
            pi: (L, K) mixing weights (softmax normalized)
            logits_list: list of K tensors, each (L, num_aa)
        """
        h = self.shared(x)
        pi = F.softmax(self.pi_head(h), dim=-1)
        logits_list = [head(h) for head in self.component_heads]

        # Mask special tokens
        if hasattr(self, 'special_token_mask'):
            penalty = float('-inf') if hard_mask else -10.0
            logits_list = [logits.masked_fill(self.special_token_mask, penalty) for logits in logits_list]

        return pi, logits_list

    def mdn_loss(self, x, target):
        """Compute MDN loss: -log(sum_k pi_k * P(target | component k))."""
        pi, logits_list = self.forward(x, hard_mask=False)

        # Stack logits: (K, L, num_aa)
        logits = torch.stack(logits_list, dim=0)

        # Compute log-probs for each component: (K, L, num_aa)
        log_probs = F.log_softmax(logits, dim=-1)

        # Get log-prob of target for each component: (K, L)
        target_expanded = target.unsqueeze(0).expand(self.K, -1)
        log_prob_target = log_probs.gather(dim=-1, index=target_expanded.unsqueeze(-1)).squeeze(-1)

        # log(sum_k pi_k * P(target|k)) = logsumexp(log(pi_k) + log(P(target|k)))
        pi_t = pi.t()  # (K, L)
        log_pi = torch.log(pi_t + 1e-10)
        log_mixture = log_pi + log_prob_target
        log_likelihood = torch.logsumexp(log_mixture, dim=0)  # (L,)

        # Negative log-likelihood per position
        nll = -log_likelihood
        return nll.mean()

    def amcl_loss(self, x, target, tau):
        """Annealed Multiple Choice Learning loss (NeurIPS 2024).

        Each of K components independently predicts the full sequence. Soft Boltzmann
        assignment starts uniform (high tau) and anneals to hard WTA (low tau).

        Loss = sum_k w_k * NLL_k, where w_k = softmax(-NLL_k / tau)
        """
        _, logits_list = self.forward(x, hard_mask=False)

        # Compute per-component NLL
        per_component_nll = []
        for logits_k in logits_list:
            log_probs_k = F.log_softmax(logits_k, dim=-1)
            nll_k = F.nll_loss(log_probs_k, target, reduction='mean')
            per_component_nll.append(nll_k)

        nlls = torch.stack(per_component_nll)

        # Boltzmann soft assignment (detached)
        with torch.no_grad():
            weights = F.softmax(-nlls / tau, dim=0)

        # Weighted loss
        loss = (weights * nlls).sum()
        return loss

    def sample(self, x, greedy=True):
        """Sample from the MDN."""
        pi, logits_list = self.forward(x, hard_mask=True)
        L = x.shape[0]
        logits = torch.stack(logits_list, dim=0)  # (K, L, num_aa)

        if greedy:
            best_k = pi.argmax(dim=-1)  # (L,)
            logits_selected = logits[best_k, torch.arange(L, device=x.device), :]
            samples = logits_selected.argmax(dim=-1)
        else:
            k_samples = torch.multinomial(pi, num_samples=1).squeeze(-1)
            logits_selected = logits[k_samples, torch.arange(L, device=x.device), :]
            probs = F.softmax(logits_selected, dim=-1)
            samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return samples

    def get_logits(self, x):
        """Get combined logits for evaluation (weighted average of components)."""
        pi, logits_list = self.forward(x, hard_mask=True)
        logits = torch.stack(logits_list, dim=0)  # (K, L, num_aa)
        # Weighted average
        pi_expanded = pi.t().unsqueeze(-1)  # (K, L, 1)
        combined_logits = (pi_expanded * logits).sum(dim=0)  # (L, num_aa)
        return combined_logits


class PosEmbedding(nn.Module):

    def __init__(self, num_embeddings):
        super(PosEmbedding, self).__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32, device=E_idx.device)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = E_idx.unsqueeze(-1) * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res


def sequential_or(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_or(res, mat)
    return res


class ProteinFeature(nn.Module):

    def __init__(self, interface_only):
        super().__init__()
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)
        self.ag_seg_id, self.hc_seg_id, self.lc_seg_id = 3, 1, 2
        self.node_pos_embedding = PosEmbedding(16)
        self.edge_pos_embedding = PosEmbedding(16)
        self.interface_only = interface_only

    def _construct_segment_ids(self, S):
        glbl_node_mask = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)
        glbl_nodes = S[glbl_node_mask]
        boa_mask, boh_mask, bol_mask = (glbl_nodes == self.boa_idx), (glbl_nodes == self.boh_idx), (glbl_nodes == self.bol_idx)
        glbl_nodes[boa_mask], glbl_nodes[boh_mask], glbl_nodes[bol_mask] = self.ag_seg_id, self.hc_seg_id, self.lc_seg_id
        segment_ids = torch.zeros_like(S)
        segment_ids[glbl_node_mask] = glbl_nodes - F.pad(glbl_nodes[:-1], (1, 0), value=0)
        segment_ids = torch.cumsum(segment_ids, dim=0)
        segment_idx = torch.zeros_like(S)
        segment_idx[glbl_node_mask] = 1.0
        segment_mask = torch.cumsum(segment_idx, dim=0)
        return segment_ids, segment_mask, torch.nonzero(segment_idx)[:, 0]

    def _radial_edges(self, X, src_dst, cutoff):
        dist = X[:, 1][src_dst]
        dist = torch.norm(dist[:, 0] - dist[:, 1], dim=-1)
        src_dst = src_dst[dist <= cutoff]
        src_dst = src_dst.transpose(0, 1)
        return src_dst

    def _knn_edges(self, X, offsets, segment_ids, is_global, top_k=5, eps=1e-6):
        for batch in range(len(offsets)):
            if batch != len(offsets) - 1:
                X_batch = X[offsets[batch]:offsets[batch+1], 1, :]
            else:
                X_batch = X[offsets[batch]:, 1, :]
            dX = torch.unsqueeze(X_batch, 0) - torch.unsqueeze(X_batch, 1)
            D = torch.sqrt(torch.sum(dX**2, 2) + eps)
            _, E_idx = torch.topk(D, top_k, dim=-1, largest=False)
            if batch == 0:
                row = torch.arange(E_idx.shape[0], device=X.device).view(-1, 1).repeat(1, top_k).view(-1)
                col = E_idx.view(-1)
            else:
                row = torch.cat([row, torch.arange(E_idx.shape[0], device=X.device).view(-1, 1).repeat(1, top_k).view(-1) + offsets[batch]], dim=0)
                col = torch.cat([col, E_idx.view(-1) + offsets[batch]], dim=0)
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges)
        ctx_edges_knn = torch.stack([row[select_edges], col[select_edges]])
        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_edges_knn = torch.stack([row[select_edges], col[select_edges]])
        return ctx_edges_knn, inter_edges_knn

    def get_node_pos(self, X, segment_mask, segment_idx):
        pos = torch.arange(X.shape[0], device=X.device) - segment_idx[segment_mask-1]
        pos_node_feats = self.node_pos_embedding(pos.view(1, X.shape[0], 1))[0, :, 0, :]
        return pos_node_feats

    def _rbf(self, D):
        D_min, D_max, D_count = 0., 20., 16
        D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
        D_mu = D_mu.view([1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def get_node_dist(self, X, eps=1e-6):
        d_NC = torch.sqrt(torch.sum((X[:, 0, :] - X[:, 1, :])**2, dim=1) + eps)
        d_CC = torch.sqrt(torch.sum((X[:, 2, :] - X[:, 1, :])**2, dim=1) + eps)
        d_OC = torch.sqrt(torch.sum((X[:, 3, :] - X[:, 1, :])**2, dim=1) + eps)
        d_NC_RBF = self._rbf(d_NC)
        d_CC_RBF = self._rbf(d_CC)
        d_OC_RBF = self._rbf(d_OC)
        dis_node_feats = torch.cat((d_NC_RBF, d_CC_RBF, d_OC_RBF), 1)
        return dis_node_feats

    def get_node_angle(self, X, segment_idx, segment_ids, eps=1e-6):
        X = X[:, :3,:].reshape(1, 3*X.shape[0], 3)
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        Dihedral_Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        cosD = (u_2*u_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.acos(cosD)
        D = F.pad(D, (1,2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        Angle_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        angle_node_feats = torch.cat((Dihedral_Angle_features, Angle_features), 2)[0]
        for i in segment_idx:
            if i == 0:
                angle_node_feats[i:i+2] = 0
            else:
                angle_node_feats[i-1:i+2] = 0
        if self.interface_only == 0:
            angle_node_feats[segment_ids == self.ag_seg_id] = 0
        return angle_node_feats

    def get_node_direct(self, Xs, segment_idx, segment_ids):
        X = Xs[:, 1,:].reshape(1, Xs.shape[0], 3)
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0,0,1,2), 'constant', 0)
        O = O.view(list(O.shape[:2]) + [3,3])
        for i in segment_idx:
            if i == 0:
                O[:, i:i+2, :, :] = 0
            else:
                O[:, i-2:i+2, :, :] = 0
        if self.interface_only == 0:
            O[:, segment_ids == self.ag_seg_id, :, :] = 0
        d_NC = (Xs[:, 0, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_NC = F.normalize(torch.matmul(O, d_NC).squeeze(-1), dim=-1)
        d_CC = (Xs[:, 2, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_CC = F.normalize(torch.matmul(O, d_CC).squeeze(-1), dim=-1)
        d_OC = (Xs[:, 3, :] - Xs[:, 1, :]).reshape(1, Xs.shape[0], 3, 1)
        d_OC = F.normalize(torch.matmul(O, d_OC).squeeze(-1), dim=-1)
        direct_node_feats = torch.cat((d_NC, d_CC, d_OC), 2)[0]
        return direct_node_feats, O

    def _quaternions(self, R):
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,:,:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)
        return Q

    def get_edge_pos(self, edge_index):
        pos = (edge_index[0:1, :] - edge_index[1:2, :]).float().unsqueeze(-1)
        pos_edge_feats = self.edge_pos_embedding(pos)[0, :, 0, :]
        return pos_edge_feats

    def get_edge_dist(self, X, edge_index, eps=1e-6):
        X_row, X_col = X[edge_index[0, :]], X[edge_index[1, :]]
        d_NC = torch.sqrt(torch.sum((X_row[:, 0, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_CAC = torch.sqrt(torch.sum((X_row[:, 1, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_CC = torch.sqrt(torch.sum((X_row[:, 2, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_OC = torch.sqrt(torch.sum((X_row[:, 3, :] - X_col[:, 1, :])**2, dim=1) + eps)
        d_NC_RBF = self._rbf(d_NC)
        d_CAC_RBF = self._rbf(d_CAC)
        d_CC_RBF = self._rbf(d_CC)
        d_OC_RBF = self._rbf(d_OC)
        dis_edge_feats = torch.cat((d_NC_RBF, d_CAC_RBF, d_CC_RBF, d_OC_RBF), 1)
        return dis_edge_feats

    def get_edge_angle(self, O, edge_index):
        O_row, O_col = O[:, edge_index[0, :], :, :].unsqueeze(2), O[:, edge_index[1, :], :, :].unsqueeze(2)
        R = torch.matmul(O_row.transpose(-1,-2), O_col)
        angle_edge_feats = self._quaternions(R)[0, :, 0, :]
        return angle_edge_feats

    def get_edge_direct(self, X, O, edge_index):
        X_row, X_col = X[edge_index[0, :]], X[edge_index[1, :]]
        _, O_col = O[:, edge_index[0, :], :, :], O[:, edge_index[1, :], :, :]
        d_NC = (X_row[:, 0, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_NC = F.normalize(torch.matmul(O_col, d_NC).squeeze(-1), dim=-1)
        d_CAC = (X_row[:, 1, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_CAC = F.normalize(torch.matmul(O_col, d_CAC).squeeze(-1), dim=-1)
        d_CC = (X_row[:, 2, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_CC = F.normalize(torch.matmul(O_col, d_CC).squeeze(-1), dim=-1)
        d_OC = (X_row[:, 3, :] - X_col[:, 1, :]).reshape(1, X_row.shape[0], 3, 1)
        d_OC = F.normalize(torch.matmul(O_col, d_OC).squeeze(-1), dim=-1)
        direct_edge_feats = torch.cat((d_NC, d_CAC, d_CC, d_OC), 2)[0]
        return direct_edge_feats

    def edge_masking(self, pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats, edge_type):
        if edge_type == 1 or edge_type == 2 or edge_type == 6 or edge_type == 7:
            pos_edge_feats *= 0
        return pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats

    @torch.no_grad()
    def construct_edges(self, X, S, batch_id):
        lengths = scatter_sum(torch.ones_like(batch_id), batch_id)
        N, max_n = batch_id.shape[0], torch.max(lengths)
        offsets = F.pad(torch.cumsum(lengths, dim=0)[:-1], pad=(1, 0), value=0)
        gni = torch.arange(N, device=batch_id.device)
        gni2lni = gni - offsets[batch_id]
        segment_ids, segment_mask, segment_idx = self._construct_segment_ids(S)
        same_bid = torch.zeros(N, max_n, device=batch_id.device)
        same_bid[(gni, lengths[batch_id] - 1)] = 1
        same_bid = 1 - torch.cumsum(same_bid, dim=-1)
        same_bid = F.pad(same_bid[:, :-1], pad=(1, 0), value=1)
        same_bid[(gni, gni2lni)] = 0
        row, col = torch.nonzero(same_bid).T
        col = col + offsets[batch_id[row]]
        is_global = sequential_or(S == self.boa_idx, S == self.boh_idx, S == self.bol_idx)
        row_global, col_global = is_global[row], is_global[col]
        not_global_edges = torch.logical_not(torch.logical_or(row_global, col_global))
        row_seg, col_seg = segment_ids[row], segment_ids[col]
        select_edges = torch.logical_and(row_seg == col_seg, not_global_edges)
        ctx_all_row, ctx_all_col = row[select_edges], col[select_edges]
        ctx_edges_rball = self._radial_edges(X, torch.stack([ctx_all_row, ctx_all_col]).T, cutoff=8.0)
        ctx_edges_knn, inter_edges_knn = self._knn_edges(X, offsets, segment_ids, is_global, top_k=8)
        if self.interface_only == 0:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 1, (row - col) == -1), select_edges, row_seg != self.ag_seg_id)
        else:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 1, (row - col) == -1), select_edges)
        ctx_edges_seq_d1 = torch.stack([row[select_edges_seq], col[select_edges_seq]])
        if self.interface_only == 0:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 2, (row - col) == -2), select_edges, row_seg != self.ag_seg_id)
        else:
            select_edges_seq = sequential_and(torch.logical_or((row - col) == 2, (row - col) == -2), select_edges)
        ctx_edges_seq_d2 = torch.stack([row[select_edges_seq], col[select_edges_seq]])
        select_edges = torch.logical_and(row_seg != col_seg, not_global_edges)
        inter_all_row, inter_all_col = row[select_edges], col[select_edges]
        inter_edges_rball = self._radial_edges(X, torch.stack([inter_all_row, inter_all_col]).T, cutoff=12.0)
        select_edges = torch.logical_and(row_seg == col_seg, torch.logical_not(not_global_edges))
        global_normal = torch.stack([row[select_edges], col[select_edges]])
        select_edges = torch.logical_and(row_global, col_global)
        global_global = torch.stack([row[select_edges], col[select_edges]])
        pos_node_feats = self.get_node_pos(X, segment_mask, segment_idx)
        dis_node_feats = self.get_node_dist(X)
        angle_node_feats = self.get_node_angle(X, segment_idx.tolist(), segment_ids)
        direct_node_feats, O = self.get_node_direct(X, segment_idx.tolist(), segment_ids)
        node_feats = torch.cat((pos_node_feats, dis_node_feats, angle_node_feats, direct_node_feats), 1)
        edges_list = [ctx_edges_rball, global_normal, global_global, ctx_edges_seq_d1, ctx_edges_knn, ctx_edges_seq_d2, inter_edges_rball, inter_edges_knn]
        edge_class_type = torch.eye(len(edges_list), dtype=torch.float, device=X.device)
        edge_feats_list = []
        for i in range(len(edges_list)):
            type_edge_feats = edge_class_type[torch.ones(edges_list[i].shape[1]).long() * i]
            pos_edge_feats = self.get_edge_pos(edges_list[i])
            dis_edge_feats = self.get_edge_dist(X, edges_list[i])
            angle_edge_feats = self.get_edge_angle(O, edges_list[i])
            direct_edge_feats = self.get_edge_direct(X, O, edges_list[i])
            pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats = self.edge_masking(pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats, i)
            edge_feats = torch.cat((type_edge_feats, pos_edge_feats, dis_edge_feats, angle_edge_feats, direct_edge_feats), 1)
            edge_feats_list.append(edge_feats)
        return edges_list, edge_feats_list, node_feats, segment_idx, segment_ids

    def forward(self, X, S, offsets):
        batch_id = torch.zeros_like(S)
        batch_id[offsets[1:-1]] = 1
        batch_id = torch.cumsum(batch_id, dim=0)
        return self.construct_edges(X, S, batch_id)


# ── MINE Estimator ────────────────────────────────────────────────────────────

class MINEEstimator(nn.Module):
    """MINE (Mutual Information Neural Estimation) - Belghazi et al., ICML 2018.

    Estimates MI lower bound via Donsker-Varadhan representation:
    MI(X,Y) >= E_joint[T(x,y)] - log(E_marginal[exp(T(x,y))])

    Stability fixes:
    - Spectral normalization on all linear layers (Lipschitz constraint)
    - Output clamping to prevent logsumexp overflow
    - EMA on marginal term (reduces variance)
    """

    def __init__(self, x_dim, y_dim, hidden_dim=128, ema_decay=0.99):
        super().__init__()
        # Spectral normalization for Lipschitz constraint
        self.statistics_net = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(x_dim + y_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            nn.utils.spectral_norm(nn.Linear(hidden_dim, 1)),
        )
        self.ema_decay = ema_decay
        self.register_buffer('ema_marginal', torch.tensor(0.0))
        self.register_buffer('ema_initialized', torch.tensor(False))

    def forward(self, x, y):
        """Compute MINE lower bound on MI(X, Y).

        Args:
            x: (B, x_dim) CDR embeddings
            y: (B, y_dim) epitope embeddings
        Returns:
            mi_estimate: scalar, MINE lower bound
        """
        perm = torch.randperm(y.size(0), device=y.device)
        y_shuffle = y[perm]

        joint = torch.cat([x, y], dim=-1)
        t_joint = self.statistics_net(joint)
        # Clamp to prevent explosion
        t_joint = torch.clamp(t_joint, -10.0, 10.0)

        marginal = torch.cat([x, y_shuffle], dim=-1)
        t_marginal = self.statistics_net(marginal)
        t_marginal = torch.clamp(t_marginal, -10.0, 10.0)

        n = t_marginal.size(0)
        # Use logsumexp with numerical stability
        log_mean_exp = torch.logsumexp(t_marginal, dim=0) - math.log(max(n, 1))

        # EMA for marginal term (reduces variance, improves stability)
        if self.training:
            exp_marginal = log_mean_exp.exp().detach()
            if not self.ema_initialized:
                self.ema_marginal = exp_marginal
                self.ema_initialized.fill_(True)
            else:
                self.ema_marginal = self.ema_decay * self.ema_marginal + (1 - self.ema_decay) * exp_marginal
            # Use EMA-corrected estimate
            mi_estimate = t_joint.mean() - self.ema_marginal.log()
        else:
            mi_estimate = t_joint.mean() - log_mean_exp

        return mi_estimate.squeeze()


# ── Latent Coupling ───────────────────────────────────────────────────────────

class AAEncoder(nn.Module):
    """Encode amino acid identity + local backbone geometry into latent space."""

    def __init__(self, n_aa=25, geom_dim=9, latent_dim=64):
        super().__init__()
        self.aa_embed = nn.Embedding(n_aa, 32)
        self.encoder = nn.Sequential(
            nn.Linear(32 + geom_dim, 128),
            nn.SiLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, aa_idx, coords):
        """
        aa_idx: (N,) amino acid indices
        coords: (N, 4, 3) backbone coords [N, CA, C, O]
        Returns: (N, latent_dim)
        """
        aa_emb = self.aa_embed(aa_idx)
        ca = coords[:, 1, :]
        rel_N = coords[:, 0, :] - ca
        rel_C = coords[:, 2, :] - ca
        rel_O = coords[:, 3, :] - ca
        geom = torch.cat([rel_N, rel_C, rel_O], dim=-1)
        return self.encoder(torch.cat([aa_emb, geom], dim=-1))


class AADecoder(nn.Module):
    """Decode latent vector to AA logits."""

    def __init__(self, latent_dim=64, n_aa=25):
        super().__init__()
        self.seq_decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.SiLU(),
            nn.Linear(128, n_aa),
        )

    def forward(self, z):
        return self.seq_decoder(z)


class AntigenGatedBottleneck(nn.Module):
    """Antigen-gated information bottleneck with gradient highway.

    FIXED ISSUES (v3):
    1. Gate too restrictive (bias=-0.8 blocked 70% signal) -> now starts at 0.0 (50%)
    2. Gradient blocking -> added residual bypass with learnable mixing
    3. Poor queries -> caller now provides position-aware queries

    Architecture:
        gate = sigmoid(W_gate @ attn_out)
        gated_cdr = cdr_h * gate
        mixed = alpha * gated_cdr + (1-alpha) * cdr_h  # Gradient highway
        output = [mixed, attn_out] -> seq_head

    The learnable alpha starts at 0.5, allowing gradients to flow through both paths.
    """

    def __init__(self, hidden_dim, init_gate_bias=0.0, init_bypass_alpha=0.5):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        # Start with neutral gate (sigmoid(0) = 0.5)
        nn.init.constant_(self.gate_proj.bias, init_gate_bias)

        # Learnable bypass mixing coefficient (gradient highway)
        # alpha=1 -> fully gated, alpha=0 -> full bypass
        self._alpha_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self.init_bypass_alpha = init_bypass_alpha

    @property
    def alpha(self):
        """Mixing coefficient: how much to use gated vs bypass."""
        # Use init value to set the operating point
        return torch.sigmoid(self._alpha_logit) * self.init_bypass_alpha * 2

    def forward(self, cdr_h, attn_out):
        """
        Args:
            cdr_h: (L, D) CDR embeddings from GNN
            attn_out: (L, D) cross-attention output from antigen

        Returns:
            (L, D*2) mixed CDR + attn_out for seq_head
        """
        gate = torch.sigmoid(self.gate_proj(attn_out))  # (L, D)
        gated_cdr = cdr_h * gate

        # Gradient highway: mix gated and ungated paths
        alpha = self.alpha
        mixed = alpha * gated_cdr + (1 - alpha) * cdr_h

        return torch.cat([mixed, attn_out], dim=-1)


class AntigenClassificationHead(nn.Module):
    """Cycle-consistent antigen classification from predicted CDR sequence.

    v32: Forces the model to actually use antigen information by requiring
    the predicted CDR to be "invertible" back to the antigen identity.

    If the model ignores antigen (predicts same CDR for all antigens),
    this loss cannot be minimized.

    Architecture:
        pred_probs: (L, n_aa) softmax probabilities from seq_head
        -> WeightedSum with learnable AA embeddings -> (L, D)
        -> MeanPool -> (D,)
        -> MLP -> (n_antigens,) logits

    During training, n_antigens = batch_size (classify which antigen in batch).
    This is a contrastive formulation that doesn't require a fixed antigen vocabulary.
    """

    def __init__(self, n_aa, hidden_dim, dropout=0.1):
        super().__init__()
        self.n_aa = n_aa
        self.hidden_dim = hidden_dim

        # Learnable AA embeddings for soft sequence encoding
        self.aa_embed = nn.Embedding(n_aa, hidden_dim)

        # Project pooled CDR representation
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def encode_soft_sequence(self, probs):
        """Encode soft sequence (probabilities) to embedding.

        Args:
            probs: (L, n_aa) softmax probabilities

        Returns:
            (hidden_dim,) pooled embedding
        """
        # Weighted sum of AA embeddings: (L, n_aa) @ (n_aa, D) -> (L, D)
        soft_emb = torch.matmul(probs, self.aa_embed.weight)
        # Mean pool over sequence length
        return soft_emb.mean(dim=0)  # (D,)

    def forward(self, cdr_probs_list, antigen_embs):
        """Compute antigen classification logits.

        Args:
            cdr_probs_list: list of (L_i, n_aa) softmax probs, one per complex
            antigen_embs: (B, D) antigen embeddings (mean-pooled from GNN)

        Returns:
            logits: (B, B) classification logits (each CDR vs all antigens)
        """
        B = len(cdr_probs_list)

        # Encode each CDR's soft sequence
        cdr_embs = []
        for probs in cdr_probs_list:
            cdr_emb = self.encode_soft_sequence(probs)
            cdr_embs.append(cdr_emb)
        cdr_embs = torch.stack(cdr_embs, dim=0)  # (B, D)

        # Project CDR embeddings
        cdr_proj = self.proj(cdr_embs)  # (B, D)

        # Compute similarity to all antigens in batch
        # (B, D) @ (D, B) -> (B, B)
        logits = torch.matmul(cdr_proj, antigen_embs.t())

        return logits


class LearnedCDRQuery(nn.Module):
    """Learned queries for cross-attention (fixes weak GNN query problem).

    Instead of using GNN output from masked CDR positions as queries,
    we learn position-aware queries that combine:
    1. Positional encoding (where in CDR)
    2. CDR type embedding (H1/H2/H3/L1/L2/L3)
    3. Global epitope context (mean-pooled antigen)

    This gives informative queries even when CDR content is unknown.
    """

    def __init__(self, hidden_dim, max_cdr_len=30, num_cdr_types=6):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Positional encoding for CDR positions
        self.pos_embed = nn.Embedding(max_cdr_len, hidden_dim)

        # CDR type embedding (H1=0, H2=1, H3=2, L1=3, L2=4, L3=5)
        self.cdr_type_embed = nn.Embedding(num_cdr_types, hidden_dim)

        # Project epitope context
        self.epitope_proj = nn.Linear(hidden_dim, hidden_dim)

        # Combine all to form query
        self.query_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, cdr_len, cdr_type_idx, epitope_context):
        """
        Args:
            cdr_len: int or tensor, length of CDR
            cdr_type_idx: int, CDR type (0-5 for H1-L3)
            epitope_context: (D,) mean-pooled epitope embedding

        Returns:
            queries: (cdr_len, D) learned query vectors
        """
        device = epitope_context.device

        # Convert tensor to int if needed
        if torch.is_tensor(cdr_len):
            cdr_len = cdr_len.item()
        cdr_len = int(cdr_len)

        # Clamp cdr_type_idx to valid range [0, 5]
        cdr_type_idx = max(0, min(5, int(cdr_type_idx)))

        # Position embeddings: (cdr_len, D)
        # Clamp positions to max_cdr_len - 1
        pos_idx = torch.arange(cdr_len, device=device).clamp(max=self.pos_embed.num_embeddings - 1)
        pos_emb = self.pos_embed(pos_idx)

        # CDR type: (1, D) -> broadcast to (cdr_len, D)
        cdr_type = self.cdr_type_embed(torch.tensor([cdr_type_idx], device=device))
        cdr_type = cdr_type.expand(cdr_len, -1)

        # Epitope context: (D,) -> (cdr_len, D)
        epi_ctx = self.epitope_proj(epitope_context).unsqueeze(0).expand(cdr_len, -1)

        # Combine and project
        combined = torch.cat([pos_emb, cdr_type, epi_ctx], dim=-1)
        queries = self.query_mlp(combined)

        return queries


# ── AntigenForce DockDesigner ─────────────────────────────────────────────────

class DockDesigner(nn.Module):
    """AntigenForce: Information-Theoretic Forcing (PLM-free).

    Four forcing mechanisms:
      1. InfoNCE: CDR must discriminate correct antigen
      2. MINE: Maximize MI(CDR, epitope)
      3. Framework dropout: Zero HC framework embeddings during training
      4. Latent coupling: CDR → latent → reconstruct AA (decodability)
    + GDPP: Spectral diversity under heavy forcing

    Forward returns 11-tuple:
      (loss, seq, coord, pairing, dock, shadow, infonce, mine, gdpp, latent_mse, latent_seq)
    """

    def __init__(self, embed_size, hidden_size, n_channel, n_layers, dropout, cdr_type, args):
        super().__init__()
        self.cdr_type = cdr_type
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = getattr(args, 'gamma', 0.5)
        self.delta = getattr(args, 'delta', 0.3)
        self.epsilon = getattr(args, 'epsilon', 0.05)
        self.dock_cutoff = getattr(args, 'dock_cutoff', 8.0)
        self.num_attn_heads = getattr(args, 'num_attn_heads', 4)
        self.n_virtual_nodes = getattr(args, 'n_virtual_nodes', 3)

        # AntigenForce-specific weights
        self.lambda_infonce = getattr(args, 'lambda_infonce', 0.5)
        self.lambda_mine = getattr(args, 'lambda_mine', 0.5)
        self.infonce_temp = getattr(args, 'infonce_temp', 0.1)
        self.p_fw_dropout = getattr(args, 'p_fw_dropout', 0.3)
        self.zeta = getattr(args, 'zeta', 0.5)
        self.latent_dim = getattr(args, 'latent_dim', 64)

        # KAN Sequence Head config
        self.use_kan_seq_head = getattr(args, 'use_kan_seq_head', False)
        self.kan_grid_size = getattr(args, 'kan_grid_size', 5)
        self.kan_spline_order = getattr(args, 'kan_spline_order', 3)

        # Hyperbolic Binding Geometry config
        self.use_hyperbolic = getattr(args, 'use_hyperbolic', False)
        self.hyperbolic_curvature = getattr(args, 'hyperbolic_curvature', 1.0)

        # v3 Fixes: Gate and Query improvements
        self.use_learned_queries = getattr(args, 'use_learned_queries', True)  # Fix weak GNN queries
        self.init_gate_bias = getattr(args, 'init_gate_bias', 0.0)  # Fix restrictive gate (was -0.8)
        self.init_bypass_alpha = getattr(args, 'init_bypass_alpha', 0.5)  # Gradient highway mixing

        # Topological Binding Signature (persistent homology)
        self.use_topo_features = getattr(args, 'use_topo_features', False)
        self.topo_dim = getattr(args, 'topo_dim', 64)
        self.lambda_topo = getattr(args, 'lambda_topo', 0.1)  # Topological loss weight

        # v4: Pairwise + Iterative Refinement (breaks conditional independence)
        self.use_pairwise = getattr(args, 'use_pairwise', False)
        self.use_iterative = getattr(args, 'use_iterative', False)
        self.n_refine_passes = getattr(args, 'n_refine_passes', 3)
        self.n_message_passes = getattr(args, 'n_message_passes', 2)
        self.pairwise_coupling_mode = getattr(args, 'pairwise_coupling_mode', 'adjacent')
        self.lambda_pairwise = getattr(args, 'lambda_pairwise', 0.1)
        self.gumbel_tau = getattr(args, 'gumbel_tau', 1.0)

        # v4: Improved MINE (hard negatives)
        self.use_improved_mine = getattr(args, 'use_improved_mine', False)
        self.mine_n_negatives = getattr(args, 'mine_n_negatives', 4)

        # v32: Direct attention + Antigen classification
        # Modes: 'direct' = attn_out only (too aggressive, loses context)
        #        'no_gate' = concat(cdr_h, attn_out) without gate (preserves context, removes dead gate)
        #        False = original gated bottleneck
        self.use_direct_attn = getattr(args, 'use_direct_attn', False)
        self.lambda_antigen_cls = getattr(args, 'lambda_antigen_cls', 0.5)  # Antigen classification loss weight

        # v32: Precomputed ESM embeddings
        self.use_esm = getattr(args, 'use_esm', False)
        self.esm_dim = getattr(args, 'esm_dim', 1280)  # ESM-2 650M default

        # MDN (Mixture Density Network) - addresses mode collapse
        self.use_mdn = getattr(args, 'use_mdn', False)
        self.mdn_k = getattr(args, 'mdn_k', 4)

        # aMCL (Annealed Multiple Choice Learning)
        self.use_amcl = getattr(args, 'use_amcl', False)
        self.amcl_tau_start = getattr(args, 'amcl_tau_start', 3.0)
        self.amcl_tau_end = getattr(args, 'amcl_tau_end', 0.05)
        self.amcl_anneal_epochs = getattr(args, 'amcl_anneal_epochs', 40)
        self._current_epoch = 0  # For AMCL tau annealing

        node_feats_mode = args.node_feats_mode
        edge_feats_mode = args.edge_feats_mode
        self.interface_only = args.interface_only

        node_feats_dim = int(node_feats_mode[0]) * 16 + int(node_feats_mode[1]) * 48 + int(node_feats_mode[2]) * 12 + int(node_feats_mode[3]) * 9
        edge_feats_dim = int(edge_feats_mode[0]) * 16 + int(edge_feats_mode[1]) * 64 + int(edge_feats_mode[2]) * 4 + int(edge_feats_mode[3]) * 12 + 8

        self.num_aa_type = len(VOCAB)
        self.mask_token_id = VOCAB.get_unk_idx()
        self.projection_head_cdr = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.SiLU(), nn.Linear(hidden_size//2, hidden_size//4))
        self.projection_head_ant = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.SiLU(), nn.Linear(hidden_size//2, hidden_size//4))

        # Rich feature encoder: 105D node_features + 3D segment_type -> embed_size (v2)
        self.use_rich_features = getattr(args, 'use_rich_features', True)
        if self.use_rich_features:
            self.protein_feature_encoder = ProteinFeatureEncoderWithSparse(
                hidden_dim=embed_size, input_dim=108, dropout=dropout,
                use_dual_path=True, fusion_type='concat')
        else:
            self.aa_embedding = nn.Embedding(self.num_aa_type, embed_size)

        # Curated epitope conditioning: zero-init so it has no effect at start
        self.epitope_embed = nn.Embedding(2, embed_size)
        nn.init.zeros_(self.epitope_embed.weight)

        # GNN: VirtualNodeEGNN
        self.gnn = VirtualNodeEGNN(
            embed_size, hidden_size, self.num_aa_type, n_channel,
            n_layers=n_layers, dropout=dropout,
            node_feats_dim=node_feats_dim, edge_feats_dim=edge_feats_dim,
            n_virtual_nodes=self.n_virtual_nodes)

        # Residue-level cross-attention (Euclidean or Hyperbolic)
        if self.use_hyperbolic:
            self.ag_cross_attn = HyperbolicAttention(
                hidden_size, num_heads=self.num_attn_heads,
                curvature=self.hyperbolic_curvature, dropout=dropout)
        else:
            self.ag_cross_attn = nn.MultiheadAttention(
                hidden_size, num_heads=self.num_attn_heads, batch_first=True)

        # Antigen-gated bottleneck: forces antigen dependence for sequence prediction
        self.antigen_gate = AntigenGatedBottleneck(
            hidden_size,
            init_gate_bias=self.init_gate_bias,
            init_bypass_alpha=self.init_bypass_alpha)

        # Learned queries for cross-attention (fixes weak GNN query problem)
        if self.use_learned_queries:
            self.learned_query = LearnedCDRQuery(hidden_size, max_cdr_len=30, num_cdr_types=6)

        # Topological Binding Signature (persistent homology)
        if self.use_topo_features:
            if not GUDHI_AVAILABLE:
                print("Warning: gudhi not installed, topological features will be placeholder")
            self.topo_encoder = TopologicalBindingSignature(
                hidden_dim=hidden_size,
                topo_dim=self.topo_dim,
                max_edge_length=self.dock_cutoff,
                use_images=False,  # Use statistics (faster, no CNN)
            )

        # v32: ESM projection layer (maps ESM dim to hidden_size)
        if self.use_esm:
            self.esm_proj = nn.Linear(self.esm_dim, hidden_size)

        # v32: Determine seq_head input dimension based on mode
        # 'direct' or True: attn_out only (hidden_size)
        # 'no_gate' or False: concat (hidden_size * 2)
        # +hidden_size if using ESM (adds esm_proj output)
        if self.use_direct_attn == 'direct' or self.use_direct_attn is True:
            seq_head_input_dim = hidden_size  # attn_out only
        else:
            seq_head_input_dim = hidden_size * 2  # cdr_h + attn_out

        if self.use_esm:
            seq_head_input_dim += hidden_size  # + esm_proj(esm_emb)

        # Seq head: v32 uses attn_out only, original uses [gated_cdr, attn_out]
        # Priority: MDN+Pairwise > Pairwise+Iterative > Pairwise > Iterative > MDN > KAN > MLP
        if self.use_mdn and self.use_pairwise:
            # MDN + Pairwise: K mixture components with pairwise message passing
            self.seq_head = MDNPairwiseHead(
                hidden_dim=seq_head_input_dim,
                num_aa=self.num_aa_type,
                K=self.mdn_k,
                dropout=dropout,
                n_message_passes=self.n_message_passes,
                lambda_pair=self.lambda_pairwise,
                vocab=VOCAB,
            )
            self.use_combined_head = False  # Uses MDN-style loss, not combined head
            self.use_mdn_pairwise = True
        elif self.use_pairwise and self.use_iterative:
            # Combined pairwise + iterative (best)
            self.seq_head = CombinedPairwiseIterativeHead(
                hidden_dim=seq_head_input_dim,
                n_aa=self.num_aa_type,
                n_passes=self.n_refine_passes,
                n_message_passes=self.n_message_passes,
                coupling_mode=self.pairwise_coupling_mode,
                lambda_pair=self.lambda_pairwise,
                gumbel_tau_init=self.gumbel_tau,
            )
            self.use_combined_head = True
            self.use_mdn_pairwise = False
        elif self.use_pairwise:
            # Pairwise only
            self.seq_head = PairwiseSequenceHead(
                hidden_dim=seq_head_input_dim,
                n_aa=self.num_aa_type,
                coupling_mode=self.pairwise_coupling_mode,
                n_message_passes=self.n_message_passes,
            )
            self.use_combined_head = False
            self.use_mdn_pairwise = False
        elif self.use_iterative:
            # Iterative only
            self.seq_head = IterativeRefinementDecoder(
                hidden_dim=seq_head_input_dim,
                n_aa=self.num_aa_type,
                n_passes=self.n_refine_passes,
                gumbel_tau_init=self.gumbel_tau,
            )
            self.use_combined_head = True
            self.use_mdn_pairwise = False
        elif self.use_mdn:
            # MDN: Mixture Density Network (addresses mode collapse)
            self.seq_head = MDNSeqHead(
                hidden_dim=seq_head_input_dim,
                num_aa=self.num_aa_type,
                K=self.mdn_k,
                dropout=dropout,
                vocab=VOCAB,
            )
            self.use_combined_head = False
            self.use_mdn_pairwise = False
        elif self.use_kan_seq_head:
            # KAN version: learnable spline functions
            self.seq_head = nn.Sequential(
                KANLinear(seq_head_input_dim, hidden_size,
                          grid_size=self.kan_grid_size,
                          spline_order=self.kan_spline_order),
                nn.Linear(hidden_size, self.num_aa_type),
            )
            self.use_combined_head = False
            self.use_mdn_pairwise = False
        else:
            # Original MLP version
            self.seq_head = nn.Sequential(
                nn.Linear(seq_head_input_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_aa_type),
            )
            self.use_combined_head = False
            self.use_mdn_pairwise = False

        # v32: Antigen classification head (cycle consistency)
        # Create if lambda_antigen_cls > 0 (works with any attention mode)
        if self.lambda_antigen_cls > 0:
            self.antigen_cls_head = AntigenClassificationHead(
                n_aa=self.num_aa_type,
                hidden_dim=hidden_size,
                dropout=dropout,
            )

        # InfoNCE projection heads
        proj_dim = hidden_size // 4
        self.epi_proj_cdr = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, proj_dim),
        )
        self.epi_proj_epitope = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, proj_dim),
        )

        # MINE estimator (v4: improved options)
        # InfoNCE is more stable than MINE (doesn't collapse to 0)
        self.mine_type = getattr(args, 'mine_type', 'infonce' if self.use_improved_mine else 'mine')
        if self.mine_type == 'infonce':
            # InfoNCE: MI >= log(N) - L_NCE (most stable)
            self.mine_estimator = InfoNCEMINE(
                x_dim=hidden_size, y_dim=hidden_size, hidden_dim=hidden_size // 2)
        elif self.mine_type == 'hard_mine':
            # HardNegativeMINE: uses hard negatives (more expressive but less stable)
            self.mine_estimator = HardNegativeMINE(
                x_dim=hidden_size, y_dim=hidden_size, hidden_dim=hidden_size // 2,
                n_negatives=self.mine_n_negatives)
        else:
            # Original MINE (unstable, collapses to 0)
            self.mine_estimator = MINEEstimator(
                x_dim=hidden_size, y_dim=hidden_size, hidden_dim=hidden_size // 2)

        # Latent coupling
        self.aa_encoder = AAEncoder(n_aa=self.num_aa_type, geom_dim=9, latent_dim=self.latent_dim)
        self.aa_decoder = AADecoder(latent_dim=self.latent_dim, n_aa=self.num_aa_type)
        self.latent_proj = nn.Linear(hidden_size, self.latent_dim)

        self.protein_feature = ProteinFeature(args.interface_only)

        # Auxiliary CDR feature reconstruction
        self.eta = getattr(args, 'eta', 0.3)
        self.cdr_feature_predictor = CDRFeaturePredictor(hidden_size, dropout=dropout)

    def seq_loss(self, _input, target):
        return F.cross_entropy(_input, target, reduction='none')

    def coord_loss(self, _input, target):
        return F.smooth_l1_loss(_input, target, reduction='sum')

    def init_mask(self, X, S, cdr_range):
        X, S, cmask = X.clone(), S.clone(), torch.zeros_like(X, device=X.device)
        n_channel, n_dim = X.shape[1:]
        for start, end in cdr_range:
            S[start:end + 1] = self.mask_token_id
            l_coord, r_coord = X[start - 1], X[end + 1]
            n_span = end - start + 2
            coord_offsets = (r_coord - l_coord).unsqueeze(0).expand(n_span - 1, n_channel, n_dim)
            coord_offsets = torch.cumsum(coord_offsets, dim=0)
            mask_coords = l_coord + coord_offsets / n_span
            X[start:end + 1] = mask_coords
            cmask[start:end + 1, ...] = 1
        return X, S, cmask

    def _contrastive_pairing_loss(self, aa_embd, S, cdr_range, segment_idx):
        aa_embd_cdr = self.projection_head_cdr(aa_embd)
        aa_embd_ant = self.projection_head_ant(aa_embd)
        cdr_logits = []
        for start, end in cdr_range:
            cdr_logits.append(torch.mean(aa_embd_cdr[start:end+1], dim=0, keepdim=True))
        cdr_logits = torch.cat(cdr_logits, dim=0)
        ant_logits = []
        segment_list = segment_idx.tolist()
        for i, index in enumerate(segment_list):
            if S[index] == VOCAB.symbol_to_idx(VOCAB.BOA):
                if index == segment_list[-1]:
                    ant_logits.append(torch.mean(aa_embd_ant[index:], dim=0, keepdim=True))
                else:
                    ant_logits.append(torch.mean(aa_embd_ant[index:segment_list[i+1]], dim=0, keepdim=True))
        if len(ant_logits) == 0 or len(ant_logits) != len(cdr_logits):
            return torch.tensor(0.0, device=aa_embd.device)
        ant_logits = torch.cat(ant_logits, dim=0)
        norm1, norm2 = cdr_logits.norm(dim=1), ant_logits.norm(dim=1)
        mat_norm = torch.einsum('i,j->ij', norm1, norm2)
        mat_sim = torch.exp(torch.einsum('ik,jk,ij->ij', cdr_logits, ant_logits, 1/mat_norm.clamp(min=1e-8)) / 1.0)
        b = cdr_logits.size(0)
        diag = mat_sim[range(b), range(b)]
        p_loss = -torch.log(diag / (mat_sim.sum(dim=1) - diag).clamp(min=1e-8)).mean()
        return p_loss

    def _dock_loss(self, Z, true_X, cdr_range, S, segment_ids):
        device = Z.device
        is_global = sequential_or(S == self.protein_feature.boa_idx,
                                  S == self.protein_feature.boh_idx,
                                  S == self.protein_feature.bol_idx)
        ag_mask = (segment_ids == self.protein_feature.ag_seg_id) & ~is_global
        ag_ca = true_X[ag_mask, 1, :]
        if ag_ca.shape[0] == 0:
            return torch.tensor(0.0, device=device)
        losses = []
        for start, end in cdr_range:
            true_cdr_ca = true_X[start:end + 1, 1, :]
            pred_cdr_ca = Z[start:end + 1, 1, :]
            true_dists = torch.cdist(ag_ca, true_cdr_ca)
            epi_mask = true_dists.min(dim=1).values < self.dock_cutoff
            if not epi_mask.any():
                continue
            epi_ca = ag_ca[epi_mask]
            pred_dists = torch.cdist(pred_cdr_ca, epi_ca)
            min_dists = pred_dists.min(dim=1).values
            losses.append(min_dists.mean())
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def _shadow_paratope_loss(self, Z, true_X, cdr_range, S, segment_ids):
        device = Z.device
        is_global = sequential_or(S == self.protein_feature.boa_idx,
                                  S == self.protein_feature.boh_idx,
                                  S == self.protein_feature.bol_idx)
        ag_mask = (segment_ids == self.protein_feature.ag_seg_id) & ~is_global
        ag_ca = true_X[ag_mask, 1, :]
        if ag_ca.shape[0] == 0:
            return torch.tensor(0.0, device=device)
        losses = []
        for start, end in cdr_range:
            true_cdr_ca = true_X[start:end + 1, 1, :]
            pred_cdr_ca = Z[start:end + 1, 1, :]
            true_d = torch.cdist(ag_ca, true_cdr_ca)
            epi_mask = true_d.min(dim=1).values < self.dock_cutoff
            if not epi_mask.any():
                continue
            epi_ca = ag_ca[epi_mask]
            true_dm = torch.cdist(true_cdr_ca, epi_ca)
            pred_dm = torch.cdist(pred_cdr_ca, epi_ca)
            losses.append((pred_dm - true_dm).abs().mean())
        if not losses:
            return torch.tensor(0.0, device=device)
        return torch.stack(losses).mean()

    def _build_masks(self, S, segment_ids, cdr_range, input_epitope_mask):
        device = S.device
        n_nodes = S.shape[0]
        is_global = sequential_or(
            S == self.protein_feature.boa_idx,
            S == self.protein_feature.boh_idx,
            S == self.protein_feature.bol_idx)
        ag_mask = (segment_ids == self.protein_feature.ag_seg_id) & ~is_global
        cdr_mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        for start, end in cdr_range:
            cdr_mask[start:end + 1] = True
        # Curated epitope from CHIMERA-Bench (intersect with ag_mask to drop globals)
        epitope_mask = input_epitope_mask.bool() & ag_mask
        return epitope_mask, cdr_mask

    def _collect_cdr_epi_embeddings(self, aa_embd, S, cdr_range, segment_ids, true_X, offsets):
        """Collect per-complex mean-pooled CDR and epitope embeddings."""
        is_global = sequential_or(S == self.protein_feature.boa_idx,
                                  S == self.protein_feature.boh_idx,
                                  S == self.protein_feature.bol_idx)
        ag_seg_id = self.protein_feature.ag_seg_id
        cdr_embds, epi_embds = [], []
        for i, (start, end) in enumerate(cdr_range):
            c_start = offsets[i].item()
            c_end = offsets[i + 1].item() if (i + 1) < len(offsets) else aa_embd.shape[0]
            cdr_emb = aa_embd[start:end + 1].mean(dim=0)
            c_seg = segment_ids[c_start:c_end]
            c_global = is_global[c_start:c_end]
            ag_local_mask = (c_seg == ag_seg_id) & ~c_global
            if not ag_local_mask.any():
                continue
            ag_global_idx = torch.where(ag_local_mask)[0] + c_start
            ag_ca = true_X[ag_global_idx, 1, :]
            ag_emb = aa_embd[ag_global_idx]
            true_cdr_ca = true_X[start:end + 1, 1, :]
            dists = torch.cdist(ag_ca, true_cdr_ca).min(dim=1).values
            epi_mask = dists < self.dock_cutoff
            if not epi_mask.any():
                continue
            epi_emb = ag_emb[epi_mask].mean(dim=0)
            cdr_embds.append(cdr_emb)
            epi_embds.append(epi_emb)
        return cdr_embds, epi_embds

    def _epitope_contrastive_loss(self, aa_embd, S, cdr_range, segment_ids, true_X, offsets):
        """InfoNCE: CDR must discriminate its correct antigen from batch negatives."""
        cdr_embds, epi_embds = self._collect_cdr_epi_embeddings(
            aa_embd, S, cdr_range, segment_ids, true_X, offsets)
        if len(cdr_embds) < 2:
            return torch.tensor(0.0, device=aa_embd.device)
        cdr_z = F.normalize(self.epi_proj_cdr(torch.stack(cdr_embds)), dim=-1)
        epi_z = F.normalize(self.epi_proj_epitope(torch.stack(epi_embds)), dim=-1)
        logits = torch.mm(cdr_z, epi_z.t()) / self.infonce_temp
        labels = torch.arange(len(cdr_embds), device=aa_embd.device)
        return F.cross_entropy(logits, labels)

    def _mine_loss(self, aa_embd, S, cdr_range, segment_ids, true_X, offsets):
        """MINE: Maximize MI(CDR, epitope). Returns negative MI."""
        cdr_embds, epi_embds = self._collect_cdr_epi_embeddings(
            aa_embd, S, cdr_range, segment_ids, true_X, offsets)
        if len(cdr_embds) < 2:
            return torch.tensor(0.0, device=aa_embd.device)
        cdr_z = torch.stack(cdr_embds)
        epi_z = torch.stack(epi_embds)
        mi_estimate = self.mine_estimator(cdr_z, epi_z)
        return -mi_estimate  # Negative: minimizing this maximizes MI

    def _gdpp_loss(self, pred_logits, true_S, cdr_range):
        """GDPP diversity loss: match eigenvalue spectra of Gram matrices."""
        device = pred_logits.device
        if len(cdr_range) < 3:
            return torch.tensor(0.0, device=device)  # Need at least 3 CDRs to avoid ill-conditioned Gram matrices
        pred_seqs, true_seqs = [], []
        offset = 0
        for start, end in cdr_range:
            cdr_len = (end - start + 1).item()
            pred_probs = F.softmax(pred_logits[offset:offset + cdr_len], dim=-1)
            true_onehot = F.one_hot(true_S[offset:offset + cdr_len], num_classes=self.num_aa_type).float()
            pred_seqs.append(pred_probs.flatten())
            true_seqs.append(true_onehot.flatten())
            offset += cdr_len
        max_len = max(s.shape[0] for s in pred_seqs)
        pred_matrix = F.normalize(torch.stack([F.pad(s, (0, max_len - s.shape[0])) for s in pred_seqs]), p=2, dim=-1)
        true_matrix = F.normalize(torch.stack([F.pad(s, (0, max_len - s.shape[0])) for s in true_seqs]), p=2, dim=-1)
        eps = 1e-2
        K_pred = pred_matrix @ pred_matrix.T + eps * torch.eye(len(pred_seqs), device=device)
        K_real = true_matrix @ true_matrix.T + eps * torch.eye(len(true_seqs), device=device)
        return F.mse_loss(torch.linalg.eigvalsh(K_pred), torch.linalg.eigvalsh(K_real))

    def _get_combined_features(self, aa_embd, S, segment_ids, offsets, cdr_range, epitope_mask, true_X=None, esm_cdr_emb=None):
        """Get combined features (gated CDR + attn + ESM) for pairwise/iterative heads.

        This extracts the features that go into the seq_head, without applying the head.
        Used by forward() to compute specialized loss for pairwise/iterative heads.
        """
        is_global = sequential_or(
            S == self.protein_feature.boa_idx,
            S == self.protein_feature.boh_idx,
            S == self.protein_feature.bol_idx)

        cdr_type_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5}
        cdr_type_idx = cdr_type_map[self.cdr_type]

        all_combined = []
        for i, (start, end) in enumerate(cdr_range):
            cdr_h = aa_embd[start:end + 1]
            cdr_len = end - start + 1
            c_start = offsets[i].item()
            c_end = offsets[i + 1].item() if i + 1 < len(offsets) else len(aa_embd)
            c_global = is_global[c_start:c_end]
            ag_idx = epitope_mask[c_start:c_end] & ~c_global
            ag_h = aa_embd[c_start:c_end][ag_idx]

            if ag_h.shape[0] > 0:
                epitope_context = ag_h.mean(dim=0)
                if self.use_learned_queries:
                    query = self.learned_query(cdr_len, cdr_type_idx, epitope_context)
                else:
                    query = cdr_h

                if self.use_hyperbolic:
                    attn_out = self.ag_cross_attn(query, ag_h, ag_h)
                else:
                    attn_out, _ = self.ag_cross_attn(
                        query.unsqueeze(0), ag_h.unsqueeze(0), ag_h.unsqueeze(0))
                    attn_out = attn_out.squeeze(0)

                if self.use_topo_features and true_X is not None:
                    cdr_coords = true_X[start:end + 1, 1, :]
                    ag_global_idx = torch.where(ag_idx)[0] + c_start
                    epi_coords = true_X[ag_global_idx, 1, :]
                    topo_cond = self.topo_encoder(cdr_coords, epi_coords)
                    attn_out = attn_out + topo_cond.unsqueeze(0).expand_as(attn_out)
            else:
                attn_out = torch.zeros_like(cdr_h)

            combined = self.antigen_gate(cdr_h, attn_out)
            cdr_len_int = cdr_len.item() if hasattr(cdr_len, 'item') else int(cdr_len)
            all_combined.append((combined, cdr_len_int))

        # Concatenate all CDR features, adding ESM if available
        if self.use_esm and esm_cdr_emb is not None:
            result = []
            esm_offset = 0
            for combined, cdr_len_int in all_combined:
                esm_slice = esm_cdr_emb[esm_offset:esm_offset + cdr_len_int]
                esm_h = self.esm_proj(esm_slice)
                result.append(torch.cat([combined, esm_h], dim=-1))
                esm_offset += cdr_len_int
            return torch.cat(result, dim=0)
        else:
            return torch.cat([c for c, _ in all_combined], dim=0)

    def _antigen_conditioned_logits(self, aa_embd, S, segment_ids, offsets, cdr_range, epitope_mask, true_X=None, esm_cdr_emb=None):
        """Antigen-conditioned sequence logits via cross-attention.

        v32 CHANGES:
        - use_direct_attn=True: seq_head reads ONLY attn_out (no gated bypass)
        - This forces ALL gradient to flow through cross-attention
        - Empirically: gate contributed 0 pp AAR (ablation: gate=0 had zero impact)
        - use_esm: adds projected ESM embeddings to the combined features

        v3 features (still supported):
        1. Learned queries (position + CDR type + epitope context)
        2. Topological binding signature (optional)
        3. Hyperbolic attention (optional)
        """
        is_global = sequential_or(
            S == self.protein_feature.boa_idx,
            S == self.protein_feature.boh_idx,
            S == self.protein_feature.bol_idx)

        # Map CDR type string to index for learned queries
        cdr_type_map = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5}
        cdr_type_idx = cdr_type_map[self.cdr_type]  # Default to H3

        all_logits = []
        esm_offset = 0  # v32: track offset into concatenated esm_cdr_emb
        for i, (start, end) in enumerate(cdr_range):
            cdr_h = aa_embd[start:end + 1]
            cdr_len = end - start + 1
            c_start = offsets[i].item()
            c_end = offsets[i + 1].item() if i + 1 < len(offsets) else len(aa_embd)
            c_global = is_global[c_start:c_end]
            ag_idx = epitope_mask[c_start:c_end] & ~c_global
            ag_h = aa_embd[c_start:c_end][ag_idx]

            if ag_h.shape[0] > 0:
                # Get epitope context for learned queries
                epitope_context = ag_h.mean(dim=0)  # (D,)

                # Use learned queries if enabled (fixes weak GNN query problem)
                if self.use_learned_queries:
                    query = self.learned_query(cdr_len, cdr_type_idx, epitope_context)
                else:
                    query = cdr_h  # Original: weak GNN output

                if self.use_hyperbolic:
                    # HyperbolicAttention: takes (L, D) directly
                    attn_out = self.ag_cross_attn(query, ag_h, ag_h)
                else:
                    # Standard MultiheadAttention: needs batched input
                    attn_out, _ = self.ag_cross_attn(
                        query.unsqueeze(0), ag_h.unsqueeze(0), ag_h.unsqueeze(0))
                    attn_out = attn_out.squeeze(0)

                # Add topological conditioning if enabled
                if self.use_topo_features and true_X is not None:
                    # Get coordinates for topological analysis
                    cdr_coords = true_X[start:end + 1, 1, :]  # CA atoms
                    ag_global_idx = torch.where(ag_idx)[0] + c_start
                    epi_coords = true_X[ag_global_idx, 1, :]

                    # Compute topological signature
                    topo_cond = self.topo_encoder(cdr_coords, epi_coords)  # (hidden_dim,)

                    # Add to attention output (broadcast across CDR length)
                    attn_out = attn_out + topo_cond.unsqueeze(0).expand_as(attn_out)
            else:
                attn_out = torch.zeros_like(cdr_h)

            # v32: Multiple attention modes
            # Empirical finding: gate=0 ablation had ZERO impact on AAR (gate is dead)
            if self.use_direct_attn == 'direct' or self.use_direct_attn is True:
                # Mode 1: attn_out only + position embeddings (aggressive, may lose context)
                device = attn_out.device
                pos_idx = torch.arange(cdr_len, device=device).clamp(max=self.learned_query.pos_embed.num_embeddings - 1)
                pos_emb = self.learned_query.pos_embed(pos_idx)  # (cdr_len, D)
                combined = attn_out + pos_emb
            elif self.use_direct_attn == 'no_gate':
                # Mode 2: Simple concat without gate (preserves context, removes dead gate)
                combined = torch.cat([cdr_h, attn_out], dim=-1)
            else:
                # Original: Antigen-gated bottleneck with gradient highway bypass
                combined = self.antigen_gate(cdr_h, attn_out)

            # v32: Add projected ESM embeddings
            if self.use_esm:
                esm_slice = esm_cdr_emb[esm_offset:esm_offset + cdr_len]
                esm_h = self.esm_proj(esm_slice)
                combined = torch.cat([combined, esm_h], dim=-1)
                esm_offset += cdr_len

            all_logits.append((combined, cdr_len))

        # Apply seq_head to all CDRs
        all_combined = torch.cat([c for c, _ in all_logits], dim=0)

        # v4: Handle pairwise/iterative heads (return logits only here, loss computed in forward())
        if hasattr(self, 'use_combined_head') and self.use_combined_head:
            # These heads have special forward() signature, use basic mode here
            if isinstance(self.seq_head, (CombinedPairwiseIterativeHead, IterativeRefinementDecoder)):
                logits = self.seq_head(all_combined)
                if isinstance(logits, tuple):
                    logits = logits[0]  # Extract logits from (logits, loss) tuple
            else:
                logits = self.seq_head(all_combined)
        else:
            # MDN: get combined logits for evaluation
            if self.use_mdn:
                logits = self.seq_head.get_logits(all_combined)
            else:
                logits = self.seq_head(all_combined)

        return logits

    def forward(self, X, S, L, offsets, node_features, segment_type,
                interface_mask, global_mask, global_types, epitope_mask,
                cdr_mask_1d=None, aux_targets=None, esm_cdr_emb=None):
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        true_X, true_S = X.clone(), S.clone()
        X, S, cmask = self.init_mask(X, S, cdr_range)
        mask = cmask[:, 0, 0].bool()
        aa_cnt = mask.sum()

        residue_mask = ~global_mask
        residue_feats = node_features[residue_mask]
        residue_seg = segment_type[residue_mask]
        residue_interface = interface_mask[residue_mask]
        H_0 = self.protein_feature_encoder(
            residue_feats, residue_seg, residue_interface, global_mask, global_types)
        H_0 = H_0 + self.epitope_embed((epitope_mask > 0.5).long())

        with torch.no_grad():
            edges_list, edge_feats_list, node_feats, segment_idx, segment_ids = self.protein_feature(X, S, offsets)

        # Framework dropout BEFORE GNN: zero out HC framework input embeddings
        # so GNN message passing propagates less framework context to CDR nodes
        if self.training and self.p_fw_dropout > 0:
            is_global = sequential_or(S == self.protein_feature.boa_idx,
                                      S == self.protein_feature.boh_idx,
                                      S == self.protein_feature.bol_idx)
            is_hc = segment_ids == self.protein_feature.hc_seg_id
            is_cdr = torch.zeros_like(mask)
            for start, end in cdr_range:
                is_cdr[start:end + 1] = True
            is_framework = is_hc & ~is_cdr & ~is_global
            fw_drop_mask = torch.rand(is_framework.sum(), device=H_0.device) < self.p_fw_dropout
            H_0 = H_0.clone()
            H_0[is_framework] = H_0[is_framework] * (~fw_drop_mask).float().unsqueeze(-1)

        epi_mask_curated, cdr_mask = self._build_masks(S, segment_ids, cdr_range, epitope_mask)
        H, Z, aa_embd = self.gnn(
            H_0, X, edges_list, edge_feats_list, node_feats, segment_ids,
            self.interface_only, epi_mask_curated, cdr_mask)

        # Seq loss via cross-attention conditioned logits
        # v4: Handle pairwise/iterative heads with specialized loss computation
        if hasattr(self, 'use_combined_head') and self.use_combined_head:
            # Get combined features first
            cdr_logits = self._antigen_conditioned_logits(
                aa_embd, S, segment_ids, offsets, cdr_range, epi_mask_curated, true_X=true_X, esm_cdr_emb=esm_cdr_emb)
            # Compute loss with pairwise/iterative head
            targets = true_S[mask]
            if isinstance(self.seq_head, CombinedPairwiseIterativeHead):
                # Re-run seq_head with targets to get specialized loss
                _, seq_loss = self.seq_head(self._get_combined_features(
                    aa_embd, S, segment_ids, offsets, cdr_range, epi_mask_curated, true_X, esm_cdr_emb=esm_cdr_emb), targets=targets)
            elif isinstance(self.seq_head, IterativeRefinementDecoder):
                _, seq_loss = self.seq_head(self._get_combined_features(
                    aa_embd, S, segment_ids, offsets, cdr_range, epi_mask_curated, true_X, esm_cdr_emb=esm_cdr_emb), targets=targets)
            else:
                seq_loss = torch.sum(self.seq_loss(cdr_logits, targets)) / aa_cnt
        else:
            cdr_logits = self._antigen_conditioned_logits(
                aa_embd, S, segment_ids, offsets, cdr_range, epi_mask_curated, true_X=true_X, esm_cdr_emb=esm_cdr_emb)
            # MDN/AMCL: use specialized loss
            if self.use_mdn:
                combined_feats = self._get_combined_features(
                    aa_embd, S, segment_ids, offsets, cdr_range, epi_mask_curated, true_X, esm_cdr_emb=esm_cdr_emb)
                if self.use_amcl:
                    # aMCL with tau annealing
                    progress = min(self._current_epoch / max(self.amcl_anneal_epochs, 1), 1.0)
                    tau = self.amcl_tau_start * (self.amcl_tau_end / self.amcl_tau_start) ** progress
                    seq_loss = self.seq_head.amcl_loss(combined_feats, true_S[mask], tau=tau)
                else:
                    # Standard MDN loss
                    seq_loss = self.seq_head.mdn_loss(combined_feats, true_S[mask])
            else:
                seq_loss = torch.sum(self.seq_loss(cdr_logits, true_S[mask])) / aa_cnt
        coord_loss = self.coord_loss(Z[mask], true_X[mask]) / aa_cnt

        if len(cdr_range) > 1 and (S == VOCAB.symbol_to_idx(VOCAB.BOA)).sum() == len(cdr_range):
            pairing_loss = self._contrastive_pairing_loss(aa_embd, S, cdr_range, segment_idx)
        else:
            pairing_loss = torch.tensor(0.0, device=X.device)

        dock_loss = self._dock_loss(Z, true_X, cdr_range, S, segment_ids)
        shadow_loss = self._shadow_paratope_loss(Z, true_X, cdr_range, S, segment_ids)

        # InfoNCE
        infonce_loss = self._epitope_contrastive_loss(
            aa_embd, S, cdr_range, segment_ids, true_X, offsets)
        # MINE
        mine_loss = self._mine_loss(
            aa_embd, S, cdr_range, segment_ids, true_X, offsets)
        # GDPP
        gdpp_loss = self._gdpp_loss(cdr_logits, true_S[mask], cdr_range)

        # Latent coupling
        z_pred = self.latent_proj(aa_embd[mask])
        with torch.no_grad():
            z_true = self.aa_encoder(true_S[mask], true_X[mask])
        latent_mse = F.mse_loss(z_pred, z_true)
        latent_logits = self.aa_decoder(z_pred)
        latent_seq_loss = F.cross_entropy(latent_logits, true_S[mask])

        # Auxiliary CDR feature reconstruction loss
        if cdr_mask_1d is not None and aux_targets is not None and cdr_mask_1d.any():
            cdr_embeddings = aa_embd[cdr_mask_1d]
            aux_loss = self.cdr_feature_predictor.aux_loss(cdr_embeddings, aux_targets)
        else:
            aux_loss = torch.tensor(0.0, device=X.device)

        # v32: Antigen classification loss (cycle consistency)
        # Forces model to actually use antigen info: pred_CDR -> antigen_id
        if self.lambda_antigen_cls > 0 and len(cdr_range) > 1:
            # Extract antigen embeddings for each complex
            is_global = sequential_or(
                S == self.protein_feature.boa_idx,
                S == self.protein_feature.boh_idx,
                S == self.protein_feature.bol_idx)
            ag_seg_id = self.protein_feature.ag_seg_id

            antigen_embs = []
            cdr_probs_list = []
            cdr_offset = 0
            for i, (start, end) in enumerate(cdr_range):
                cdr_len = end - start + 1
                # Get antigen embedding for this complex
                c_start = offsets[i].item()
                c_end = offsets[i + 1].item() if i + 1 < len(offsets) else len(aa_embd)
                c_seg = segment_ids[c_start:c_end]
                c_global = is_global[c_start:c_end]
                ag_idx = (c_seg == ag_seg_id) & ~c_global
                ag_h = aa_embd[c_start:c_end][ag_idx]
                if ag_h.shape[0] > 0:
                    antigen_embs.append(ag_h.mean(dim=0))
                else:
                    antigen_embs.append(torch.zeros(aa_embd.shape[-1], device=aa_embd.device))
                # Get CDR softmax probabilities
                cdr_logits_i = cdr_logits[cdr_offset:cdr_offset + cdr_len]
                cdr_probs_list.append(F.softmax(cdr_logits_i, dim=-1))
                cdr_offset += cdr_len

            antigen_embs = torch.stack(antigen_embs, dim=0)  # (B, D)
            # Compute contrastive antigen classification
            ag_cls_logits = self.antigen_cls_head(cdr_probs_list, antigen_embs)  # (B, B)
            targets = torch.arange(len(cdr_range), device=X.device)
            antigen_cls_loss = F.cross_entropy(ag_cls_logits, targets)
        else:
            antigen_cls_loss = torch.tensor(0.0, device=X.device)

        loss = (seq_loss
                + self.alpha * coord_loss
                + self.beta * pairing_loss
                + self.gamma * dock_loss
                + self.delta * shadow_loss
                + self.lambda_infonce * infonce_loss
                + self.lambda_mine * mine_loss
                + self.epsilon * gdpp_loss
                + self.zeta * (latent_mse + latent_seq_loss)
                + self.eta * aux_loss
                + self.lambda_antigen_cls * antigen_cls_loss)

        return (loss, seq_loss, coord_loss, pairing_loss, dock_loss, shadow_loss,
                infonce_loss, mine_loss, gdpp_loss, latent_mse, latent_seq_loss, aux_loss,
                antigen_cls_loss)

    def set_epoch(self, epoch):
        """Set current epoch for AMCL tau annealing."""
        self._current_epoch = epoch

    def anneal_gumbel_temperature(self, factor=0.99, min_tau=0.1):
        """Anneal Gumbel-softmax temperature for iterative refinement.

        Should be called once per epoch by the trainer.
        """
        if hasattr(self, 'use_combined_head') and self.use_combined_head:
            if hasattr(self.seq_head, 'anneal_temperature'):
                self.seq_head.anneal_temperature(factor=factor, min_tau=min_tau)
            elif hasattr(self.seq_head, 'gumbel_tau'):
                self.seq_head.gumbel_tau = max(min_tau, self.seq_head.gumbel_tau * factor)

    def generate(self, X, S, L, offsets, node_features, segment_type,
                 interface_mask, global_mask, global_types, epitope_mask, greedy=True, esm_cdr_emb=None):
        cdr_range = torch.tensor(
            [(cdr.index(self.cdr_type), cdr.rindex(self.cdr_type)) for cdr in L],
            dtype=torch.long, device=X.device
        ) + offsets[:-1].unsqueeze(-1)

        true_X, true_S = X.clone(), S.clone()
        X, S, cmask = self.init_mask(X, S, cdr_range)
        mask = cmask[:, 0, 0].bool()
        aa_cnt = mask.sum()

        special_mask = torch.tensor(VOCAB.get_special_mask(), device=S.device, dtype=torch.long)
        smask = special_mask.repeat(aa_cnt, 1).bool()

        residue_mask = ~global_mask
        residue_feats = node_features[residue_mask]
        residue_seg = segment_type[residue_mask]
        residue_interface = interface_mask[residue_mask]
        H_0 = self.protein_feature_encoder(
            residue_feats, residue_seg, residue_interface, global_mask, global_types)
        H_0 = H_0 + self.epitope_embed((epitope_mask > 0.5).long())

        with torch.no_grad():
            edges_list, edge_feats_list, node_feats, segment_idx, segment_ids = self.protein_feature(X, S, offsets)

        epi_mask_curated, cdr_mask = self._build_masks(S, segment_ids, cdr_range, epitope_mask)
        H, Z, aa_embd = self.gnn(
            H_0, X, edges_list, edge_feats_list, node_feats, segment_ids,
            self.interface_only, epi_mask_curated, cdr_mask)

        X = X.clone()
        X[mask] = Z[mask]

        logits = self._antigen_conditioned_logits(
            aa_embd, S, segment_ids, offsets, cdr_range, epi_mask_curated, true_X=true_X, esm_cdr_emb=esm_cdr_emb)
        logits = logits.masked_fill(smask, float('-inf'))

        if greedy:
            S[mask] = torch.argmax(logits, dim=-1)
        else:
            prob = F.softmax(logits, dim=-1)
            S[mask] = torch.multinomial(prob, num_samples=1).squeeze()
        snll_all = self.seq_loss(logits, S[mask])

        return snll_all, S, X, true_X, cdr_range, logits

    def infer(self, batch, device, greedy=True):
        X, S, L, offsets = batch['X'].to(device), batch['S'].to(device), batch['L'], batch['offsets'].to(device)

        node_features = batch['node_features'].to(device)
        segment_type = batch['segment_type'].to(device)
        interface_mask = batch['interface_mask'].to(device)
        epitope_mask = batch['epitope_mask'].to(device)
        global_mask = batch['global_mask'].to(device)
        global_types = batch['global_types'].to(device)

        # v32: ESM embeddings (precomputed)
        esm_cdr_emb = batch.get('esm_cdr_emb')
        if esm_cdr_emb is not None:
            esm_cdr_emb = esm_cdr_emb.to(device)

        snll_all, pred_S, pred_X, true_X, cdr_range, logits = self.generate(
            X, S, L, offsets, node_features, segment_type,
            interface_mask, global_mask, global_types, epitope_mask,
            greedy=greedy, esm_cdr_emb=esm_cdr_emb)

        pred_S, cdr_range = pred_S.tolist(), cdr_range.tolist()
        pred_X, true_X = pred_X.detach().cpu().numpy(), true_X.detach().cpu().numpy()

        seq, x, true_x, logits_list = [], [], [], []
        logit_offset = 0
        for start, end in cdr_range:
            end = end + 1
            length = end - start
            seq.append(''.join([VOCAB.idx_to_symbol(pred_S[i]) for i in range(start, end)]))
            x.append(pred_X[start:end])
            true_x.append(true_X[start:end])
            logits_list.append(logits[logit_offset:logit_offset + length].detach().cpu())
            logit_offset += length

        ppl = [0 for _ in range(len(cdr_range))]
        lens = [0 for _ in ppl]
        offset = 0
        for i, (start, end) in enumerate(cdr_range):
            length = end - start + 1
            for t in range(length):
                ppl[i] += snll_all[t + offset]
            offset += length
            lens[i] = length

        ppl = [p / n for p, n in zip(ppl, lens)]
        ppl = torch.exp(torch.tensor(ppl, device=device)).tolist()

        return ppl, seq, x, true_x, logits_list
