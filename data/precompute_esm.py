"""Precompute ESM-2 embeddings for CDR positions.

Runs frozen ESM-2 once per complex with CDR positions masked,
saves embeddings to avoid runtime computation during training.

Usage:
    python precompute_esm.py --esm_model esm2_t33_650M_UR50D
    python precompute_esm.py --esm_model esm2_t30_150M_UR50D --batch_size 8  # faster, smaller
"""
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm

import torch
import esm

# Add paths
_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR / 'code'))
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent.parent))  # parent genbio directory

from shared.features_v2 import V2FeatureLoader

# ESM-2 model configs: (repr_layer, embedding_dim)
ESM_CONFIGS = {
    'esm2_t6_8M_UR50D': (6, 320),
    'esm2_t12_35M_UR50D': (12, 480),
    'esm2_t30_150M_UR50D': (30, 640),
    'esm2_t33_650M_UR50D': (33, 1280),
    'esm2_t36_3B_UR50D': (36, 2560),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--esm_model', default='esm2_t33_650M_UR50D',
                   choices=list(ESM_CONFIGS.keys()))
    p.add_argument('--features_path', default='./data/processed/rich_features_v2.pt')
    p.add_argument('--output_dir', default='./data/processed/esm_embeddings')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--cdr_types', default='1,2,3', help='Comma-separated CDR types to precompute')
    return p.parse_args()


def get_cdr_positions(cdr_mask, cdr_type_idx, n_heavy):
    """Get CDR positions within heavy chain for a given CDR type."""
    # cdr_mask: (N, 6) where columns are H1,H2,H3,L1,L2,L3
    # cdr_type_idx: 0=H1, 1=H2, 2=H3
    positions = []
    for i in range(n_heavy):
        if cdr_mask[i, cdr_type_idx] > 0.5:
            positions.append(i)
    return positions


@torch.no_grad()
def compute_esm_embeddings(
    esm_model, esm_alphabet, esm_repr_layer,
    hc_sequence, cdr_positions, device
):
    """Compute ESM-2 embeddings for CDR positions with masking.

    Args:
        esm_model: Frozen ESM-2 model
        esm_alphabet: ESM-2 alphabet for tokenization
        esm_repr_layer: Which layer to extract representations from
        hc_sequence: Heavy chain sequence string (without CDR masked)
        cdr_positions: List of CDR position indices within HC
        device: torch device

    Returns:
        esm_cdr_emb: (L_cdr, esm_dim) tensor of ESM embeddings at CDR positions
    """
    batch_converter = esm_alphabet.get_batch_converter()

    # Build masked sequence: replace CDR positions with placeholder
    hc_chars = list(hc_sequence)
    for pos in cdr_positions:
        hc_chars[pos] = 'A'  # placeholder, will be ESM-masked

    # Tokenize
    _, _, tokens = batch_converter([("hc", ''.join(hc_chars))])
    tokens = tokens.to(device)

    # Apply mask token at CDR positions (+1 for <cls> token)
    for pos in cdr_positions:
        tokens[0, pos + 1] = esm_alphabet.mask_idx

    # Run ESM-2
    results = esm_model(tokens, repr_layers=[esm_repr_layer])
    representations = results["representations"][esm_repr_layer]

    # Extract CDR positions (skip <cls> and <eos>)
    # representations shape: (1, seq_len+2, esm_dim)
    hc_repr = representations[0, 1:len(hc_sequence)+1, :]  # (L_hc, esm_dim)

    return hc_repr[cdr_positions]  # (L_cdr, esm_dim)


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load ESM-2
    print(f'Loading {args.esm_model}...')
    esm_repr_layer, esm_dim = ESM_CONFIGS[args.esm_model]
    esm_loader = getattr(esm.pretrained, args.esm_model)
    esm_model, esm_alphabet = esm_loader()
    esm_model = esm_model.to(device)
    esm_model.eval()
    print(f'ESM-2 loaded: {esm_dim}D embeddings from layer {esm_repr_layer}')

    # Load features
    print(f'Loading features from {args.features_path}...')
    feature_loader = V2FeatureLoader(args.features_path)
    complex_ids = feature_loader.complex_ids
    print(f'Found {len(complex_ids)} complexes')

    # Create output directory
    output_dir = Path(args.output_dir) / args.esm_model
    output_dir.mkdir(parents=True, exist_ok=True)

    cdr_types = [int(x) for x in args.cdr_types.split(',')]
    print(f'Computing embeddings for CDR types: {cdr_types}')

    # Process each CDR type
    for cdr_type in cdr_types:
        cdr_type_idx = cdr_type - 1  # 0-indexed
        cdr_name = f'H{cdr_type}'
        output_file = output_dir / f'esm_cdr{cdr_type}.pt'

        if output_file.exists():
            print(f'{output_file} already exists, skipping...')
            continue

        print(f'\nProcessing {cdr_name}...')
        embeddings = {}

        for cid in tqdm(complex_ids, desc=cdr_name):
            feats = feature_loader.get_features(cid)
            n_heavy = feats['n_heavy']
            sequence = feats['sequence']
            cdr_mask = feats['cdr_mask']

            hc_sequence = sequence[:n_heavy]
            cdr_positions = get_cdr_positions(cdr_mask, cdr_type_idx, n_heavy)

            if not cdr_positions:
                embeddings[cid] = None
                continue

            embeddings[cid] = compute_esm_embeddings(
                esm_model, esm_alphabet, esm_repr_layer,
                hc_sequence, cdr_positions, device
            ).cpu()

        # Save
        torch.save({
            'embeddings': embeddings,
            'esm_model': args.esm_model,
            'esm_dim': esm_dim,
            'cdr_type': cdr_type,
        }, output_file)

        n_valid = sum(1 for v in embeddings.values() if v is not None)
        print(f'Saved {output_file}: {n_valid}/{len(embeddings)} valid embeddings')

    print('\nDone!')


if __name__ == '__main__':
    main()
