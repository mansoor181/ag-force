"""Feature dataset for HierAb (P4).

Uses pre-computed v2 105D features per residue instead of learned embeddings.
"""
import os
import sys
import json
import torch
import numpy as np
from typing import List, Dict, Optional

# Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = os.path.dirname(_THIS_DIR)
_GENBIO_ROOT = os.path.normpath(os.path.join(_CODE_DIR, '..', '..'))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)
if _GENBIO_ROOT not in sys.path:
    sys.path.insert(0, _GENBIO_ROOT)

from shared.features_v2 import V2FeatureLoader
from shared.sparse_features import extract_aux_targets, mask_cdr_features
from data.pdb_utils import VOCAB


class FeatureDataset(torch.utils.data.Dataset):
    """Dataset using pre-computed v2 features.

    Loads from both:
    - CHIMERA complex_features for structure (X) and labels (L)
    - v2_features.pt for pre-computed per-residue features
    - (optional) ESM embeddings for CDR positions
    """

    def __init__(
        self,
        complex_ids: List[str],
        v2_features_path: str,
        complex_features_dir: str,
        cdr_type: str = '3',  # H1='1', H2='2', H3='3'
        numbering_scheme: str = 'imgt',
        esm_embeddings_path: Optional[str] = None,  # v32: precomputed ESM
    ):
        """
        Args:
            complex_ids: List of complex IDs to include
            v2_features_path: Path to v2_features.pt
            complex_features_dir: Path to complex_features/*.pt directory
            cdr_type: CDR type to design ('1', '2', or '3')
            numbering_scheme: 'imgt' or 'chothia'
            esm_embeddings_path: Path to precomputed ESM embeddings (e.g., esm_cdr3.pt)
        """
        super().__init__()
        self.complex_ids = complex_ids
        self.cdr_type = cdr_type
        self.numbering_scheme = numbering_scheme
        self.complex_features_dir = complex_features_dir

        # Load v2 features (v2: 105D)
        self.feature_loader = V2FeatureLoader(v2_features_path)

        # v32: Load precomputed ESM embeddings
        self.esm_embeddings = None
        self.esm_dim = 0
        if esm_embeddings_path and os.path.exists(esm_embeddings_path):
            esm_data = torch.load(esm_embeddings_path, map_location='cpu')
            self.esm_embeddings = esm_data['embeddings']
            self.esm_dim = esm_data['esm_dim']
            print(f"Loaded ESM embeddings: {self.esm_dim}D from {esm_embeddings_path}")

        # Filter to available complexes
        self.available_ids = [
            cid for cid in complex_ids
            if cid in self.feature_loader
        ]
        print(f"FeatureDataset: {len(self.available_ids)}/{len(complex_ids)} complexes available")

    def __len__(self):
        return len(self.available_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        complex_id = self.available_ids[idx]

        # Load v2 features
        feats = self.feature_loader.get_features(complex_id)

        # Load complex_features for CDR labels and structure validation
        cf_path = os.path.join(self.complex_features_dir, f'{complex_id}.pt')
        complex_data = torch.load(cf_path, map_location='cpu')

        n_heavy = feats['n_heavy']
        n_light = feats['n_light']
        n_antigen = feats['n_antigen']

        # Build X: [N+3, 4, 3] with global tokens
        # Insert global tokens at chain boundaries
        atom14 = feats['atom14_coords']  # (N, 14, 3)
        ca_coords = feats['ca_coords']   # (N, 3)

        # Extract backbone: N, CA, C, O (indices 0, 1, 2, 3 in atom14)
        X_residues = atom14[:, :4, :]  # (N, 4, 3)

        # Center coordinates on CA centroid (critical for GNN stability)
        # Store centroid to uncenter predicted coords during evaluation
        ca_centroid = ca_coords.mean(dim=0)  # (3,)
        X_residues = X_residues - ca_centroid.unsqueeze(0).unsqueeze(1)  # (N, 4, 3)

        # Build with global tokens
        # Format: [BOH, H_residues..., BOL, L_residues..., BOA, AG_residues...]
        n_total = X_residues.shape[0] + 3  # +3 for global tokens
        X = torch.zeros(n_total, 4, 3, dtype=torch.float32)
        S = torch.zeros(n_total, dtype=torch.long)
        node_features = torch.zeros(n_total, 105, dtype=torch.float32)
        segment_type = torch.zeros(n_total, 3, dtype=torch.float32)
        interface_mask = torch.zeros(n_total, dtype=torch.float32)
        epitope_mask = torch.zeros(n_total, dtype=torch.float32)

        # Map positions
        # BOH at 0, H residues at 1:n_heavy+1
        # BOL at n_heavy+1, L residues at n_heavy+2:n_heavy+n_light+2
        # BOA at n_heavy+n_light+2, AG residues at n_heavy+n_light+3:end
        h_start, h_end = 1, 1 + n_heavy
        l_start, l_end = h_end + 1, h_end + 1 + n_light
        a_start, a_end = l_end + 1, l_end + 1 + n_antigen

        # Fill residue data
        X[h_start:h_end] = X_residues[:n_heavy]
        X[l_start:l_end] = X_residues[n_heavy:n_heavy+n_light]
        X[a_start:a_end] = X_residues[n_heavy+n_light:]

        # Global token coords = mean of chain
        X[0] = X[h_start:h_end].mean(dim=0) if n_heavy > 0 else torch.zeros(4, 3)
        X[h_end] = X[l_start:l_end].mean(dim=0) if n_light > 0 else torch.zeros(4, 3)
        X[l_end] = X[a_start:a_end].mean(dim=0) if n_antigen > 0 else torch.zeros(4, 3)

        # Fill node features (105D) - zeros for global tokens
        node_features[h_start:h_end] = feats['node_features'][:n_heavy]
        node_features[l_start:l_end] = feats['node_features'][n_heavy:n_heavy+n_light]
        node_features[a_start:a_end] = feats['node_features'][n_heavy+n_light:]

        # Fill interface_mask (antibody-side flag for sparse complementarity routing)
        im = feats['interface_mask']
        interface_mask[h_start:h_end] = im[:n_heavy]
        interface_mask[l_start:l_end] = im[n_heavy:n_heavy+n_light]
        interface_mask[a_start:a_end] = im[n_heavy+n_light:]

        # Fill epitope_mask (antigen-side curated epitope labels from CHIMERA-Bench)
        em = feats['epitope_mask']
        epitope_mask[h_start:h_end] = em[:n_heavy]
        epitope_mask[l_start:l_end] = em[n_heavy:n_heavy+n_light]
        epitope_mask[a_start:a_end] = em[n_heavy+n_light:]

        # Fill segment type
        segment_type[0:h_end, 0] = 1.0       # Heavy
        segment_type[h_end:l_end, 1] = 1.0   # Light
        segment_type[l_end:a_end, 2] = 1.0   # Antigen

        # Build S (sequence indices)
        seq = feats['sequence']
        S[0] = VOCAB.symbol_to_idx(VOCAB.BOH)
        S[h_end] = VOCAB.symbol_to_idx(VOCAB.BOL)
        S[l_end] = VOCAB.symbol_to_idx(VOCAB.BOA)

        for i, aa in enumerate(seq[:n_heavy]):
            S[h_start + i] = VOCAB.symbol_to_idx(aa)
        for i, aa in enumerate(seq[n_heavy:n_heavy+n_light]):
            S[l_start + i] = VOCAB.symbol_to_idx(aa)
        for i, aa in enumerate(seq[n_heavy+n_light:]):
            S[a_start + i] = VOCAB.symbol_to_idx(aa)

        # Build CDR label string L
        # Format: '0' for non-CDR, '1' for H1, '2' for H2, '3' for H3
        cdr_mask = feats['cdr_mask']  # (N, 6): H1,H2,H3,L1,L2,L3
        L = ['0'] * n_total

        # Map CDR positions (only H1, H2, H3 for now, indexed as 1, 2, 3)
        for i in range(n_heavy):
            for cdr_idx in range(3):  # H1, H2, H3
                if cdr_mask[i, cdr_idx] > 0.5:
                    L[h_start + i] = str(cdr_idx + 1)
                    break

        L = ''.join(L)

        # Global mask for ProteinFeatureEncoder
        global_mask = torch.zeros(n_total, dtype=torch.bool)
        global_mask[0] = True       # BOH
        global_mask[h_end] = True   # BOL
        global_mask[l_end] = True   # BOA

        global_types = torch.tensor([0, 1, 2], dtype=torch.long)  # BOH, BOL, BOA

        # Extract true CDR sequence for evaluation
        cdr_idx = int(self.cdr_type) - 1  # 0, 1, or 2 for H1, H2, H3
        heavy_seq = seq[:n_heavy]
        cdr_positions = [i for i in range(n_heavy) if cdr_mask[i, cdr_idx] > 0.5]
        true_cdr_seq = ''.join([heavy_seq[i] for i in cdr_positions]) if cdr_positions else ''

        # CDR range in the X tensor (for coordinate extraction)
        cdr_start = h_start + cdr_positions[0] if cdr_positions else h_start
        cdr_end = h_start + cdr_positions[-1] + 1 if cdr_positions else h_start

        # Build CDR mask for the target CDR (1D bool over n_total)
        cdr_mask_1d = torch.zeros(n_total, dtype=torch.bool)
        for i in cdr_positions:
            cdr_mask_1d[h_start + i] = True

        # Extract auxiliary reconstruction targets BEFORE masking
        aux_targets = extract_aux_targets(node_features, cdr_mask_1d)

        # Mask all features for CDR positions (prevents information leakage)
        mask_cdr_features(node_features, cdr_mask_1d)

        # v32: Get precomputed ESM embeddings for this complex's CDR
        esm_cdr_emb = None
        if self.esm_embeddings is not None:
            esm_cdr_emb = self.esm_embeddings[complex_id]

        return {
            'X': X,
            'S': S,
            'L': L,
            'node_features': node_features,
            'segment_type': segment_type,
            'interface_mask': interface_mask,
            'epitope_mask': epitope_mask,
            'global_mask': global_mask,
            'global_types': global_types,
            'complex_id': complex_id,
            'true_cdr_seq': true_cdr_seq,
            'cdr_range': (cdr_start, cdr_end),
            'cdr_mask_1d': cdr_mask_1d,
            'aux_targets': aux_targets,
            'ca_centroid': ca_centroid,
            'esm_cdr_emb': esm_cdr_emb,  # v32: (L_cdr, esm_dim) or None
        }

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch with variable-length sequences."""
        Xs, Ss, Ls = [], [], []
        node_feats_list, seg_type_list = [], []
        interface_mask_list = []
        epitope_mask_list = []
        global_masks, global_types_list = [], []
        cdr_mask_1d_list, aux_targets_list = [], []
        complex_ids, true_cdr_seqs, cdr_ranges = [], [], []
        ca_centroids = []
        esm_cdr_emb_list = []  # v32: ESM embeddings
        offsets = [0]

        for data in batch:
            Xs.append(data['X'])
            Ss.append(data['S'])
            Ls.append(data['L'])
            node_feats_list.append(data['node_features'])
            seg_type_list.append(data['segment_type'])
            interface_mask_list.append(data['interface_mask'])
            epitope_mask_list.append(data['epitope_mask'])
            global_masks.append(data['global_mask'])
            global_types_list.append(data['global_types'])
            cdr_mask_1d_list.append(data['cdr_mask_1d'])
            aux_targets_list.append(data['aux_targets'])
            complex_ids.append(data['complex_id'])
            true_cdr_seqs.append(data['true_cdr_seq'])
            cdr_ranges.append(data['cdr_range'])
            ca_centroids.append(data['ca_centroid'])
            if data['esm_cdr_emb'] is not None:
                esm_cdr_emb_list.append(data['esm_cdr_emb'])
            offsets.append(offsets[-1] + len(data['S']))

        result = {
            'X': torch.cat(Xs, dim=0),
            'S': torch.cat(Ss, dim=0),
            'L': Ls,
            'node_features': torch.cat(node_feats_list, dim=0),
            'segment_type': torch.cat(seg_type_list, dim=0),
            'interface_mask': torch.cat(interface_mask_list, dim=0),
            'epitope_mask': torch.cat(epitope_mask_list, dim=0),
            'global_mask': torch.cat(global_masks, dim=0),
            'global_types': torch.cat(global_types_list, dim=0),
            'cdr_mask_1d': torch.cat(cdr_mask_1d_list, dim=0),
            'aux_targets': torch.cat(aux_targets_list, dim=0),
            'offsets': torch.tensor(offsets, dtype=torch.long),
            'complex_ids': complex_ids,
            'true_cdr_seqs': true_cdr_seqs,
            'cdr_ranges': cdr_ranges,
            'ca_centroids': ca_centroids,
        }

        # v32: Concatenate ESM embeddings if present
        if esm_cdr_emb_list:
            result['esm_cdr_emb'] = torch.cat(esm_cdr_emb_list, dim=0)  # (total_cdr_len, esm_dim)

        return result


def load_split_ids(
    split_name: str,
    splits_dir: str = './data/splits',
    available_ids: Optional[List[str]] = None,
    random_seed: int = 42,
) -> Dict[str, List[str]]:
    """Load train/val/test IDs from split JSON or create random split.

    Args:
        split_name: One of 'epitope_group', 'antigen_fold', 'temporal', or 'random'
        splits_dir: Directory containing split JSON files
        available_ids: For 'random' split, list of available complex IDs
        random_seed: Random seed for 'random' split

    Returns:
        Dict with 'train', 'val', 'test' lists
    """
    # Use shared utility
    from shared.features_v2 import load_split_ids as _load_split_ids
    return _load_split_ids(
        split_name=split_name,
        splits_dir=splits_dir,
        available_ids=available_ids,
        random_seed=random_seed,
    )
