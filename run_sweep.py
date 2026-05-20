"""WandB sweep runner for AntigenForce (P4) v4.

Thin wrapper that applies sweep hyperparameters to the config,
then calls chimera_trainer's run_pipeline (train + test + evaluate).

Usage:
    # Create sweep and run agents on multiple GPUs:
    bash launch_sweep.sh --gpus 0,1 --count 20

    # Or manually:
    wandb sweep sweep.yaml
    python run_sweep.py --gpu 0 --sweep_id <id> --count 10

    # Single trial with specific params (for debugging):
    python run_sweep.py --gpu 0 --single --lr 2e-4 --dropout 0.15
"""

import argparse
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from chimera_trainer import load_config, build_config, run_pipeline

log = logging.getLogger(__name__)

# Parameters that can be swept (must match sweep.yaml)
SWEEP_PARAMS_FLOAT = [
    "lr", "weight_decay", "dropout", "anneal_base",
    "alpha", "beta", "gamma", "delta", "epsilon",
    "lambda_infonce", "infonce_temp", "p_fw_dropout", "zeta", "eta",
    "lambda_pairwise", "gumbel_tau", "lambda_mine",
]
SWEEP_PARAMS_INT = [
    "batch_size", "n_layers", "num_attn_heads", "latent_dim",
    "n_refine_passes", "n_message_passes",
]
SWEEP_PARAMS_BOOL = [
    "use_iterative", "use_pairwise", "use_improved_mine",
]
SWEEP_PARAMS_STR = [
    "mine_type",
]
SWEEP_PARAMS = SWEEP_PARAMS_FLOAT + SWEEP_PARAMS_INT + SWEEP_PARAMS_BOOL + SWEEP_PARAMS_STR


def build_sweep_config(sweep_params, gpu=0):
    """Build config namespace from base config + sweep overrides."""
    cfg, shared = load_config()

    mock_args = SimpleNamespace(
        split="epitope_group", cdr_type="3", gpu=gpu, seed=42,
        lr=None, batch_size=None, max_epoch=None,
        no_wandb=True,
        test_only=False, checkpoint=None, run_name=None,
        # Loss weights
        alpha=None, beta=None, gamma=None, delta=None, epsilon=None,
        dock_cutoff=None, num_attn_heads=None,
        # AntigenForce core
        lambda_infonce=None, lambda_mine=None, infonce_temp=None,
        p_fw_dropout=None, zeta=None, latent_dim=None, eta=None,
        # v4: Iterative refinement
        use_iterative=None, n_refine_passes=None, gumbel_tau=None,
        # v4: Pairwise coupling
        use_pairwise=None, n_message_passes=None, lambda_pairwise=None,
        # v4: Improved MINE
        use_improved_mine=None, mine_type=None, mine_n_negatives=None,
        # v3 flags (defaults)
        use_learned_queries=None, init_gate_bias=None, init_bypass_alpha=None,
        # Disabled features
        use_kan_seq_head=None, kan_grid_size=None,
        use_hyperbolic=None, hyperbolic_curvature=None,
        use_topo_features=None,
    )
    c = build_config(cfg, shared, mock_args)

    # Apply sweep overrides
    for key, val in sweep_params.items():
        if hasattr(c, key):
            setattr(c, key, val)

    return c


def run_trial(c, wandb_run):
    """Run one full trial: train + test + evaluation."""
    c.run_name = f"antigenforce_sweep_{wandb_run.id}"
    full_summary = run_pipeline(c, wandb_run)
    return full_summary


def sweep_train_fn():
    """Entry point called by wandb agent."""
    import wandb

    run = wandb.init()
    params = dict(wandb.config)
    gpu = params.pop("gpu", 0)

    c = build_sweep_config(params, gpu=gpu)
    run_trial(c, run)
    run.finish()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="AntigenForce v4 sweep runner")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--count", type=int, default=20, help="Number of sweep trials")
    parser.add_argument("--sweep_id", type=str, default=None,
                        help="Existing sweep ID to join as agent")
    parser.add_argument("--single", action="store_true",
                        help="Run a single trial with specified params (no sweep)")

    # Add CLI args for all sweep params
    for k in SWEEP_PARAMS_FLOAT:
        parser.add_argument(f"--{k}", type=float, default=None)
    for k in SWEEP_PARAMS_INT:
        parser.add_argument(f"--{k}", type=int, default=None)
    for k in SWEEP_PARAMS_BOOL:
        parser.add_argument(f"--{k}", action="store_true", default=None)
    for k in SWEEP_PARAMS_STR:
        parser.add_argument(f"--{k}", type=str, default=None)

    args = parser.parse_args()

    import wandb

    if args.single:
        params = {}
        for k in SWEEP_PARAMS:
            val = getattr(args, k, None)
            if val is not None:
                params[k] = val
        c = build_sweep_config(params, gpu=args.gpu)

        run = wandb.init(
            project="antigenforce",
            name="antigenforce_single_trial",
            config=vars(c),
        )
        c.run_name = f"antigenforce_single_{run.id}"
        run_pipeline(c, run)
        run.finish()
        return

    # Sweep agent mode
    sweep_cfg_path = _SCRIPT_DIR / "sweep.yaml"
    with open(sweep_cfg_path) as f:
        sweep_config = yaml.safe_load(f)

    sweep_config["parameters"]["gpu"] = {"value": args.gpu}

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep_config, project="antigenforce")
        log.info("Created sweep: %s", sweep_id)

    wandb.agent(sweep_id, function=sweep_train_fn, count=args.count,
                project="antigenforce")


if __name__ == "__main__":
    main()
