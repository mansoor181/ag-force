"""HierAb Trainer (Hydra-based).

HierAb (v32): Hierarchical Antibody Design with Cycle-Consistent Antigen Reconstruction

Key features:
  1. Direct attention: seq_head reads attn_out only (forces cross-attn gradient flow)
  2. Antigen classification: cycle consistency loss (pred_CDR -> antigen_id)
  3. ESM-2 embeddings: precomputed CDR position embeddings
  4. Slower LR decay: anneal_base=0.97 (prevents premature convergence)

Usage:
    python trainer.py
    python trainer.py training.gpu=1 training.max_epoch=50
    python trainer.py model.use_esm=false
"""

import datetime
import glob
import os
import sys
import time
from types import SimpleNamespace

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE_DIR = _SCRIPT_DIR
_GENBIO_ROOT = os.path.normpath(os.path.join(_CODE_DIR, '..', '..'))
_CHIMERA_ROOT = os.path.normpath(os.path.join(_GENBIO_ROOT, '..'))

sys.path.insert(0, _CODE_DIR)
sys.path.insert(0, _GENBIO_ROOT)
sys.path.insert(0, _CHIMERA_ROOT)
sys.path.insert(0, os.path.join(_CHIMERA_ROOT, "baselines"))

from model.core import DockDesigner
from data.dataset import FeatureDataset
from data.pdb_utils import VOCAB

from shared.features_v2 import get_available_complex_ids, load_split_ids
from chimera_utils import (
    load_shared_config, EarlyStopping, ModelCheckpoint,
    setup_wandb, seed_everything, to_device, save_predictions,
    run_full_evaluation, FULL_METRIC_KEYS,
)
from benchmark.evaluation.metrics import aar as chimera_aar, kabsch_rmsd, tm_score as chimera_tm_score, count_liabilities

METRIC_KEYS = ["ppl", "aar", "rmsd", "tm_score", "n_liabilities", "top3_aar", "top5_aar", "top3_caar", "top5_caar"]
LOSS_KEYS = ["loss", "seq", "coord", "pair", "dock", "shadow", "infonce", "mine", "gdpp", "latent_mse", "latent_seq", "aux", "antigen_cls"]


def build_model_args(cfg: DictConfig) -> SimpleNamespace:
    """Convert Hydra config to SimpleNamespace for model compatibility."""
    m = cfg.model
    t = cfg.training
    d = cfg.dataset
    return SimpleNamespace(
        embed_size=m.embed_size,
        hidden_size=m.hidden_size,
        n_layers=m.n_layers,
        dropout=m.dropout,
        num_attn_heads=m.num_attn_heads,
        n_virtual_nodes=m.n_virtual_nodes,
        alpha=m.alpha,
        beta=m.beta,
        gamma=m.gamma,
        delta=m.delta,
        epsilon=m.epsilon,
        dock_cutoff=m.dock_cutoff,
        use_direct_attn=m.use_direct_attn,
        lambda_antigen_cls=m.lambda_antigen_cls,
        lambda_infonce=m.lambda_infonce,
        lambda_mine=m.lambda_mine,
        infonce_temp=m.infonce_temp,
        zeta=m.zeta,
        latent_dim=m.latent_dim,
        p_fw_dropout=m.p_fw_dropout,
        eta=m.eta,
        node_feats_mode=str(m.node_feats_mode),
        edge_feats_mode=str(m.edge_feats_mode),
        interface_only=m.interface_only,
        mode=str(m.mode),
        use_esm=m.use_esm,
        esm_dim=m.esm_dim,
        esm_model=m.esm_model,
        use_learned_queries=m.use_learned_queries,
        init_gate_bias=m.init_gate_bias,
        init_bypass_alpha=m.init_bypass_alpha,
        use_kan_seq_head=m.use_kan_seq_head,
        kan_grid_size=m.kan_grid_size,
        kan_spline_order=m.kan_spline_order,
        use_hyperbolic=m.use_hyperbolic,
        hyperbolic_curvature=m.hyperbolic_curvature,
        use_topo_features=m.use_topo_features,
        topo_dim=m.topo_dim,
        lambda_topo=m.lambda_topo,
        use_pairwise=m.use_pairwise,
        use_iterative=m.use_iterative,
        n_refine_passes=m.n_refine_passes,
        n_message_passes=m.n_message_passes,
        pairwise_coupling_mode=m.pairwise_coupling_mode,
        lambda_pairwise=m.lambda_pairwise,
        gumbel_tau=m.gumbel_tau,
        use_improved_mine=m.use_improved_mine,
        mine_type=m.mine_type,
        mine_n_negatives=m.mine_n_negatives,
        use_rich_features=True,
        anneal_base=t.anneal_base,
    )


def find_features(data_root="./data"):
    """Find the latest v2 features file."""
    final = os.path.join(data_root, 'processed/rich_features_v2.pt')
    if os.path.exists(final):
        return final
    ckpts = sorted(glob.glob(os.path.join(data_root, 'processed/checkpoints_v2/ckpt_*.pt')))
    if ckpts:
        return ckpts[-1]
    raise FileNotFoundError("No v2 features found")


def _agg(m):
    return {k: float(np.mean(v)) for k, v in m.items()}


def train_epoch(model, loader, opt, device, grad_clip):
    model.train()
    ms = {k: [] for k in LOSS_KEYS}
    for b in loader:
        b = to_device(b, device)
        vals = model(b["X"], b["S"], b["L"], b["offsets"],
                     b["node_features"], b["segment_type"],
                     b["interface_mask"], b["global_mask"], b["global_types"], b["epitope_mask"],
                     cdr_mask_1d=b.get("cdr_mask_1d"), aux_targets=b.get("aux_targets"),
                     esm_cdr_emb=b.get("esm_cdr_emb"))
        opt.zero_grad()
        vals[0].backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        for k, v in zip(LOSS_KEYS, vals):
            ms[k].append(v.item() if torch.is_tensor(v) else v)
    return _agg(ms)


def valid_epoch(model, loader, device):
    model.eval()
    ms = {k: [] for k in LOSS_KEYS}
    with torch.no_grad():
        for b in loader:
            b = to_device(b, device)
            vals = model(b["X"], b["S"], b["L"], b["offsets"],
                         b["node_features"], b["segment_type"],
                         b["interface_mask"], b["global_mask"], b["global_types"], b["epitope_mask"],
                         esm_cdr_emb=b.get("esm_cdr_emb"))
            for k, v in zip(LOSS_KEYS, vals):
                ms[k].append(v.item() if torch.is_tensor(v) else v)
    return _agg(ms)


def compute_topk_metrics(logits, true_seq, contact_positions=None, k_values=(3, 5)):
    """Compute Top-K AAR and CAAR from logits.

    Args:
        logits: (L, n_aa) tensor of logits
        true_seq: str, ground truth sequence
        contact_positions: list of contact position indices for CAAR (0-indexed within CDR)
        k_values: tuple of K values to compute

    Returns:
        dict with top{k}_aar and top{k}_caar for each k
    """
    results = {}
    L = len(true_seq)
    if logits.shape[0] != L:
        return {f'top{k}_aar': 0.0 for k in k_values} | {f'top{k}_caar': 0.0 for k in k_values}

    true_indices = [VOCAB.symbol_to_idx(aa) for aa in true_seq]

    for k in k_values:
        topk_indices = torch.topk(logits, k, dim=-1).indices  # (L, k)
        in_topk = torch.tensor([
            true_indices[pos] in topk_indices[pos].tolist() if true_indices[pos] is not None else False
            for pos in range(L)
        ])
        results[f"top{k}_aar"] = in_topk.float().mean().item()

        # Compute CAAR (Contact AAR) - only at contact positions
        if contact_positions and len(contact_positions) > 0:
            contact_in_topk = [in_topk[i].item() for i in contact_positions if i < L]
            if contact_in_topk:
                results[f'top{k}_caar'] = sum(contact_in_topk) / len(contact_in_topk)
            else:
                results[f'top{k}_caar'] = results[f'top{k}_aar']
        else:
            results[f'top{k}_caar'] = results[f'top{k}_aar']

    return results


def build_contact_cache(features_dir, cdr_type):
    """Build cache mapping complex_id -> list of contact position indices.

    Contact positions = CDR positions that are in the paratope (binding interface).
    Uses IMGT numbering to find CDR positions:
      H1: 27-38, H2: 56-65, H3: 105-117
      L1: 27-38, L2: 56-65, L3: 105-117
    """
    contact_cache = {}
    cdr_label = f"H{cdr_type}"
    chain_type = 'heavy' if cdr_label.startswith('H') else 'light'

    # IMGT CDR boundaries
    cdr_ranges = {
        'H1': (27, 38), 'H2': (56, 65), 'H3': (105, 117),
        'L1': (27, 38), 'L2': (56, 65), 'L3': (105, 117),
    }
    cdr_start, cdr_end = cdr_ranges.get(cdr_label, (0, 0))

    if not features_dir or not os.path.isdir(features_dir):
        return contact_cache

    for pt_file in os.listdir(features_dir):
        if not pt_file.endswith('.pt'):
            continue
        cid = pt_file.replace('.pt', '')
        try:
            feat = torch.load(os.path.join(features_dir, pt_file), map_location='cpu', weights_only=False)
            numbering = feat.get('numbering', {}).get('imgt', {}).get(chain_type, [])
            if not numbering:
                continue

            # Get target chain letter from complex_id (format: pdb_heavy_light_antigen)
            parts = cid.split('_')
            heavy_chain = parts[1] if len(parts) > 1 else ''
            light_chain = parts[2] if len(parts) > 2 else ''
            target_chain = heavy_chain if chain_type == 'heavy' else light_chain

            # Find CDR positions in the numbering array
            cdr_positions = []
            for i, entry in enumerate(numbering):
                if isinstance(entry, tuple) and len(entry) >= 1:
                    resid = entry[0]
                    if cdr_start <= resid <= cdr_end:
                        cdr_positions.append((i, resid))

            if not cdr_positions:
                continue

            # Get paratope residues for this chain
            paratope = feat.get('paratope_residues', [])
            paratope_resids = {resid for chain, resid, aa in paratope if chain == target_chain}

            # Find contact positions (0-indexed within CDR)
            contact_pos = []
            for cdr_idx, (seq_pos, resid) in enumerate(cdr_positions):
                if resid in paratope_resids:
                    contact_pos.append(cdr_idx)

            contact_cache[cid] = contact_pos
        except Exception:
            pass

    return contact_cache


def run_inference(model, loader, device, cdr_type, contact_cache=None):
    model.eval()
    preds, metrics = [], {k: [] for k in METRIC_KEYS}
    cdr_label = f"H{cdr_type}"
    contact_cache = contact_cache or {}
    with torch.no_grad():
        for b in loader:
            ppls, seqs, xs, true_xs, logits_list = model.infer(b, device)
            ca_centroids = b['ca_centroids']
            for i in range(len(b["L"])):
                cid, true_seq = b['complex_ids'][i], b['true_cdr_seqs'][i]
                pred_ca = (xs[i][:, 1, :] if xs[i].ndim == 3 else xs[i])
                true_ca = (true_xs[i][:, 1, :] if true_xs[i].ndim == 3 else true_xs[i])
                if isinstance(pred_ca, torch.Tensor):
                    pred_ca = pred_ca.cpu().numpy()
                if isinstance(true_ca, torch.Tensor):
                    true_ca = true_ca.cpu().numpy()
                centroid = ca_centroids[i]
                centroid_np = centroid.cpu().numpy() if isinstance(centroid, torch.Tensor) else centroid
                pred_ca = pred_ca + centroid_np
                true_ca = true_ca + centroid_np
                aar = chimera_aar(seqs[i], true_seq)
                rmsd, tm = kabsch_rmsd(pred_ca, true_ca), chimera_tm_score(pred_ca, true_ca)
                liab = count_liabilities(seqs[i])
                contact_pos = contact_cache.get(cid, None)
                topk = compute_topk_metrics(logits_list[i], true_seq, contact_pos)
                top3_aar, top5_aar = topk["top3_aar"], topk["top5_aar"]
                top3_caar, top5_caar = topk["top3_caar"], topk["top5_caar"]
                for k, v in zip(METRIC_KEYS, [ppls[i], aar, rmsd, tm, liab, top3_aar, top5_aar, top3_caar, top5_caar]):
                    metrics[k].append(v)
                preds.append({
                    "complex_id": cid, "cdr_type": cdr_label,
                    "pred_sequence": seqs[i], "true_sequence": true_seq,
                    "pred_coords": pred_ca, "true_coords": true_ca,
                    "ppl": ppls[i], "aar": aar, "rmsd": rmsd, "tm_score": tm, "n_liabilities": liab,
                    "top3_aar": top3_aar, "top5_aar": top5_aar, "top3_caar": top3_caar, "top5_caar": top5_caar
                })
    return preds, {k: float(np.mean(v)) for k, v in metrics.items() if v}


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    args = build_model_args(cfg)
    args.cdr_type = str(cfg.dataset.cdr_type)
    args.split = cfg.dataset.split
    args.data_root = cfg.data_root
    args.output_dir = cfg.results_dir
    args.numbering_scheme = cfg.numbering_scheme
    args.features_path = cfg.features_path
    args.esm_embeddings_dir = cfg.dataset.esm_embeddings_dir

    args.seed = cfg.training.seed
    args.gpu = cfg.training.gpu
    args.lr = cfg.training.lr
    args.weight_decay = cfg.training.weight_decay
    args.batch_size = cfg.training.batch_size
    args.max_epoch = cfg.training.max_epoch
    args.grad_clip = cfg.training.grad_clip
    args.num_workers = cfg.training.num_workers
    args.run_name = cfg.training.run_name
    args.patience = cfg.callbacks.early_stopping.patience

    args.use_wandb = cfg.wandb.enabled
    args.wandb_project = cfg.wandb.project

    args.test_only = getattr(cfg.training, 'test_only', False)
    args.checkpoint = getattr(cfg.training, 'checkpoint', None)

    run_pipeline(args)


def run_pipeline(c, wandb_run=None):
    seed_everything(c.seed)
    device = torch.device(f"cuda:{c.gpu}" if torch.cuda.is_available() else "cpu")
    cdr_label = f"H{c.cdr_type}"

    if c.run_name:
        run_name = c.run_name
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"hierab_cdr{c.cdr_type}_{c.split}_{timestamp}"

    save_dir = os.path.join(c.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    features_path = find_features(c.data_root)
    cf_dir = os.path.join(c.data_root, 'processed/complex_features')
    available = get_available_complex_ids()
    splits = load_split_ids(c.split, available_ids=available if c.split == 'random' else None, random_seed=c.seed)

    esm_emb_path = None
    if c.use_esm:
        esm_emb_path = os.path.join(c.esm_embeddings_dir, c.esm_model, f"esm_cdr{c.cdr_type}.pt")
        if not os.path.exists(esm_emb_path):
            print(f"Warning: ESM embeddings not found at {esm_emb_path}, disabling ESM")
            esm_emb_path = None
            c.use_esm = False

    train_set = FeatureDataset(splits['train'], features_path, cf_dir, c.cdr_type, c.numbering_scheme, esm_embeddings_path=esm_emb_path)
    valid_set = FeatureDataset(splits['val'], features_path, cf_dir, c.cdr_type, c.numbering_scheme, esm_embeddings_path=esm_emb_path)
    test_set = FeatureDataset(splits['test'], features_path, cf_dir, c.cdr_type, c.numbering_scheme, esm_embeddings_path=esm_emb_path)
    print(f"Data: {len(train_set)}/{len(valid_set)}/{len(test_set)} train/val/test")

    model = DockDesigner(c.embed_size, c.hidden_size, 4, n_layers=c.n_layers, dropout=c.dropout, cdr_type=c.cdr_type, args=c).to(device)
    print(f"HierAb: {sum(p.numel() for p in model.parameters()):,} params")

    if c.use_wandb and not getattr(c, 'test_only', False):
        wandb_run = setup_wandb(c.wandb_project, run_name, vars(c), enabled=True)

    if getattr(c, 'test_only', False):
        ckpt_path = c.checkpoint or os.path.join(save_dir, "checkpoints", "best.pt")
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    else:
        train_loader = DataLoader(train_set, c.batch_size, shuffle=True, num_workers=c.num_workers, collate_fn=FeatureDataset.collate_fn)
        valid_loader = DataLoader(valid_set, c.batch_size, shuffle=False, num_workers=c.num_workers, collate_fn=FeatureDataset.collate_fn)
        opt = torch.optim.AdamW(model.parameters(), lr=float(c.lr), weight_decay=c.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda e: c.anneal_base ** e)
        es, ckpt = EarlyStopping(patience=c.patience, mode="min"), ModelCheckpoint(save_dir, mode="min")

        for ep in range(c.max_epoch):
            t0 = time.time()
            # Set epoch for AMCL tau annealing
            if hasattr(model, 'set_epoch'):
                model.set_epoch(ep)
            tm, vm = train_epoch(model, train_loader, opt, device, c.grad_clip), valid_epoch(model, valid_loader, device)
            sched.step()
            _, vm2 = run_inference(model, valid_loader, device, c.cdr_type)
            cur_lr = opt.param_groups[0]["lr"]
            best = ckpt.save(model, opt, sched, ep, vm["loss"])
            print(f"Ep {ep:3d} | loss={tm['loss']:.4f}/{vm['loss']:.4f} "
                  f"seq={tm['seq']:.3f}/{vm['seq']:.3f} coord={tm['coord']:.3f}/{vm['coord']:.3f} "
                  f"dock={tm['dock']:.3f}/{vm['dock']:.3f} shad={tm['shadow']:.3f}/{vm['shadow']:.3f} "
                  f"aar={vm2.get('aar',0):.3f} rmsd={vm2.get('rmsd',0):.2f} "
                  f"lr={cur_lr:.6f} {'*' if best else ''} [{time.time()-t0:.0f}s]")
            if wandb_run:
                log_dict = {"epoch": ep, "lr": cur_lr}
                log_dict.update({f"train_{k}": tm[k] for k in LOSS_KEYS})
                log_dict.update({f"val_{k}": vm[k] for k in LOSS_KEYS})
                log_dict.update({f"val_{k}": v for k, v in vm2.items()})
                wandb_run.log(log_dict)

            if hasattr(model, 'anneal_gumbel_temperature'):
                model.anneal_gumbel_temperature(factor=0.99, min_tau=0.1)

            if es(vm["loss"]):
                break
        ckpt.load_best(model, device)

    test_loader = DataLoader(test_set, c.batch_size, shuffle=False, num_workers=c.num_workers, collate_fn=FeatureDataset.collate_fn)
    complex_features_dir = os.path.join(c.data_root, "processed", "complex_features")
    contact_cache = build_contact_cache(complex_features_dir, c.cdr_type)
    preds, topk_summary = run_inference(model, test_loader, device, c.cdr_type, contact_cache)
    pred_dir = os.path.join(save_dir, "predictions", cdr_label)
    save_predictions(preds, pred_dir)
    if c.split != 'random':
        _, summary, _ = run_full_evaluation(pred_dir, c.split, c.data_root, cdr_type_hint=cdr_label, numbering_scheme=c.numbering_scheme)
        # Merge Top-K metrics back
        summary.update({k: topk_summary[k] for k in ['top3_aar', 'top5_aar', 'top3_caar', 'top5_caar'] if k in topk_summary})
    else:
        summary = topk_summary

    def fmt(v):
        return f"{v['mean']:.3f}" if isinstance(v, dict) else f"{v:.3f}"
    print(f"\nTest ({cdr_label}): " + " ".join(f"{k}={fmt(v)}" for k, v in summary.items()))

    if wandb_run:
        for k in FULL_METRIC_KEYS:
            v = summary.get(k, {})
            if isinstance(v, dict):
                v = v.get("mean", 0)
            wandb_run.log({f"test_{cdr_label}_{k}": v})
        wandb_run.log({"test_n": len(preds)})
        wandb_run.finish()

    return summary


if __name__ == "__main__":
    main()
