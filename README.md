# v32: Cycle-Consistent Antigen Reconstruction + Direct Attention + ESM-2

## Empirical Motivation

v32 is based on rigorous empirical diagnosis of AntigenForce (p4), which revealed:

1. **Antigen-swap test**: When replacing antigen with a random one, predictions have **0.69 similarity** (should be ~0 if model uses antigen). AAR drops only **1.7 pp**.

2. **Gate ablation**: Setting `gate=0` (kill gated path) causes **zero AAR change**. The gated path contributes nothing. Setting `gate=1` actually *helps* (+0.4 pp).

3. **Cross-attention ablation**: Zeroing `attn_out` costs only **1.0 pp** AAR. The entire antigen pathway contributes ~2 pp total.

4. **MI objectives disconnect**: MINE reaches -1.13 nats, InfoNCE reaches 0.88 -- both optimized, but val_aar stays flat after epoch 8. The MI gradients flow into embeddings; prediction reads from cross-attention -- different pathways.

5. **Vocab collapse**: For CDR length=12 across 36 different antigens, predictions use only **3.9 unique AAs/position** vs true **11.25**. Model predicts marginal P(AA|position).

## v32 Key Changes

### 1. Attention Modes (Three Options)

Based on ablation: gate=0 had ZERO impact (gate is dead). Three modes available:

```python
# 'no_gate' (RECOMMENDED): Remove dead gate, keep concat
combined = torch.cat([cdr_h, attn_out], dim=-1)  # (L, 512)

# 'direct': Aggressive, attn_out only (lost too much context in testing)
combined = attn_out + pos_emb  # (L, 256) - needs position info

# False: Original gated bottleneck (gate does nothing empirically)
combined = self.antigen_gate(cdr_h, attn_out)  # (L, 512)
```

**`no_gate` mode** preserves the original capacity while removing the dead gate mechanism.

### 2. Antigen Classification Head (Cycle Consistency)

New head that takes predicted CDR softmax probabilities and classifies which antigen they were designed for:

```python
class AntigenClassificationHead(nn.Module):
    """Cycle consistency: if model ignores antigen, all CDRs look similar,
    and antigen classification becomes impossible."""

    def forward(self, cdr_probs_list, antigen_embs):
        # Encode soft sequences via weighted sum of AA embeddings
        cdr_embs = [self.encode_soft_sequence(p) for p in cdr_probs_list]
        # Project and compute similarity to all antigens in batch
        logits = matmul(proj(stack(cdr_embs)), antigen_embs.T)
        return logits  # (B, B) contrastive classification
```

The loss is `CrossEntropy(logits, arange(B))` -- each CDR should classify its own antigen.

**Key insight**: This loss flows *through* the seq_head output. If the model collapses to identical predictions for all antigens, classification accuracy = 1/B (random). The only way to minimize this loss is to produce antigen-specific CDRs.

### 3. Disable Inert Losses

MINE, InfoNCE, and latent coupling are disabled (`lambda=0`). Empirically:
- MINE loss optimized (-1.13 nats) but val_aar unchanged
- InfoNCE loss optimized (0.88) but val_aar unchanged
- These losses operate on GNN embeddings, but prediction reads cross-attention -- different pathways

### 4. Slower LR Decay

`anneal_base: 0.97` (was 0.9). At epoch 30:
- Old: lr = 1e-3 * 0.9^30 ≈ 4e-5 (too small to escape local optima)
- New: lr = 1e-3 * 0.97^30 ≈ 4e-4 (10x higher)

### 5. Precomputed ESM-2 Embeddings

v32 adds frozen ESM-2 (650M) embeddings for CDR positions:

```python
# Precompute embeddings (run once)
python precompute_esm.py --esm_model esm2_t33_650M_UR50D

# Embeddings saved to: ./data/processed/esm_embeddings/esm2_t33_650M_UR50D/esm_cdr{1,2,3}.pt
```

The embeddings are computed with CDR positions masked (replaced with `<mask>` token), then extracted at CDR positions. This provides rich evolutionary context without information leakage.

**Integration**:
- `esm_proj`: Linear(1280, hidden_size) projects ESM embeddings
- Combined features: `concat(cdr_h, attn_out, esm_h)` -> seq_head
- Memory efficient: ~2.5GB GPU savings vs runtime ESM forward pass

## Architecture

```
Input: Ab-Ag complex
  |
  v
ProteinFeatureEncoder (105D + 3D segment) -> H_0
  |
  v
Framework Dropout (p=0.3, training only)
  |
  v
VirtualNodeEGNN (4 layers, 3 virtual nodes)
  |
  v
aa_embd (GNN output for all residues)
  |
  +---> cdr_h (CDR positions from GNN)
  |
  +---> LearnedCDRQuery (pos_emb + cdr_type + epitope_context)
  |       |
  |       v
  |     Cross-Attention(query, ag_h, ag_h) -> attn_out
  |
  +---> ESM-2 (frozen, precomputed) -> esm_cdr_emb -> esm_proj -> esm_h
  |
  v
[no_gate mode] concat(cdr_h, attn_out, esm_h) -> (L, 768)
  |
  v
seq_head(combined) -> logits
  |
  +---> softmax(logits) -> probs
  |       |
  |       v
  |     AntigenClassificationHead(probs, antigen_embs) -> ag_cls_loss [NEW]
  |
  v
Losses: seq + coord + pairing + dock + shadow + gdpp + aux + antigen_cls
```

## Diagnostic Targets

Run antigen-swap experiment after training to measure:

| Metric | AntigenForce (baseline) | v32 Target |
|--------|------------------------|------------|
| swap similarity | 0.69 | < 0.40 |
| AAR drop on wrong antigen | 1.7 pp | > 5 pp |
| val_aar | 0.36 | > 0.40 |

If swap similarity drops significantly, conditioning works and AAR should follow.

## Usage

```bash
# Step 1: Precompute ESM embeddings (run once, takes ~30 min)
python precompute_esm.py --esm_model esm2_t33_650M_UR50D --cdr_types 1,2,3

# Step 2: Train
# Quick test
python chimera_trainer.py --max_epoch 2 --no-wandb

# Full training
python chimera_trainer.py --split epitope_group --cdr_type 3

# Test only (load checkpoint)
python chimera_trainer.py --test_only --checkpoint /path/to/best.pt
```

## Config

Key v32 settings in `config.yaml`:
```yaml
dataset:
  esm_model: esm2_t33_650M_UR50D
  esm_embeddings_dir: ./data/processed/esm_embeddings

model:
  # Attention mode: 'no_gate' (recommended), 'direct', or false
  use_direct_attn: no_gate    # Remove dead gate, keep concat(cdr_h, attn_out)
  lambda_antigen_cls: 0.5     # antigen classification weight (cycle consistency)
  lambda_infonce: 0.0         # disabled (empirically inert)
  lambda_mine: 0.0            # disabled (empirically inert)
  zeta: 0.0                   # latent coupling disabled
  # ESM-2 embeddings
  use_esm: true               # Enable precomputed ESM embeddings
  esm_dim: 1280               # ESM-2 650M dimension

training:
  anneal_base: 0.97           # slower LR decay (0.9 -> 0.97)
```

## Files

- `code/models.py` - DockDesigner with `use_direct_attn`, `AntigenClassificationHead`, and ESM projection
- `code/v2_dataset.py` - Dataset with ESM embeddings loading
- `config.yaml` - v32 defaults (direct_attn on, MI losses off, ESM on)
- `chimera_trainer.py` - 13-tuple loss handling, ESM integration
- `precompute_esm.py` - One-time ESM-2 embedding precomputation

## Expected Outcome

If v32 works:
- Antigen-swap similarity should drop from 0.69 to <0.4
- AAR should improve from 0.36 to >0.40
- Per-position vocab diversity should increase

If v32 doesn't help:
- Next step: set-prediction loss (6b) to break per-position CE ceiling
- Or: discriminative pretraining (6c) if GNN representation itself lacks antigen info
