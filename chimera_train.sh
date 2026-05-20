#!/bin/bash
# AntigenForce training script (Hydra-based)
# MDN + AMCL for mode collapse prevention
# Usage: bash chimera_train.sh --gpu 0 --epochs 50

# ==============================================================================
# TRAINING CONFIG - Optimized hyperparameters
# ==============================================================================

# Training (from sweep)
gpu=0
max_epoch=100
batch_size=4
lr=2.29e-4
weight_decay=0.032
anneal_base=0.954
grad_clip=0.5
num_workers=4
seed=42

# Dataset
split="epitope_group"   # epitope_group | antigen_fold | temporal | random
cdr_type="1"            # 1 (H1) | 2 (H2) | 3 (H3)

# Model architecture (from sweep)
embed_size=128
hidden_size=256
n_layers=5
dropout=0.1
num_attn_heads=4
n_virtual_nodes=3

# Loss weights - REBALANCED based on loss landscape analysis
# Original sweep optimized for STRUCTURE, not SEQUENCE
# Key changes:
#   - gamma=0: dock loss doesn't learn, wastes 28% of gradient
#   - alpha reduced: coord dominates and overfits (1.5x val/train gap)
#   - zeta reduced: latent competes with main seq path
#   - beta/eta=0: pairing saturates, aux not helping

alpha=1.3         # coord (reduced from 1.301 - was overfitting)
beta=0            # pairing (saturates quickly, disabled)
gamma=0           # dock (NOT LEARNING - disabled until fixed)
delta=0.664         # shadow (reduced from 0.664)
epsilon=0.05      # GDPP (reduced, saturates)
eta=0             # aux (disabled)
zeta=0          # latent (reduced from 0.674)
dock_cutoff=8.0

# v32: Attention mode + Antigen classification
# gated attention
use_direct_attn=false       # no_gate | direct | false (sweep used false)
lambda_antigen_cls=0.2      # down from 0.2


infonce_temp=0.119
latent_dim=64

# Framework dropout
p_fw_dropout=0.3 # reduced from 0.3

# ESM-2 embeddings 
use_esm=true
esm_dim=1280
esm_model="esm2_t33_650M_UR50D"

# Architectural choices
use_learned_queries=false
init_gate_bias=0.0
init_bypass_alpha=0.5

# Pairwise/Iterative
# When use_mdn=true AND use_pairwise=true: uses MDNPairwiseHead (K components + J couplings)
# When use_pairwise=true AND use_iterative=true (no mdn): uses CombinedPairwiseIterativeHead
use_pairwise=true     # Enable pairwise J couplings
use_iterative=false   # MUST be false to use MDNPairwiseHead
n_refine_passes=2
n_message_passes=2    # Message passing rounds in pairwise
pairwise_coupling_mode="adjacent"
gumbel_tau=0.641
lambda_pairwise=0.3   # pairwise energy weight in MDNPairwiseHead


# Information-theoretic losses (reduced - they saturate quickly)
use_improved_mine=false
mine_type="infonce"
mine_n_negatives=4
lambda_infonce=0   # reduced from 0.2 (saturates by epoch 20)
lambda_mine=0      # reduced


# MDN (Mixture Density Network) + Pairwise
# When both use_mdn=true and use_pairwise=true: MDNPairwiseHead
# K components, each with its own J matrix for local AA constraints
use_mdn=true
mdn_k=2               # number of mixture components

# aMCL (Annealed Multiple Choice Learning)
# Faster annealing based on loss landscape analysis
use_amcl=true
amcl_tau_start=1.5    # lower start for faster specialization
amcl_tau_end=0.2      # don't go too low (avoid collapse)
amcl_anneal_epochs=15 # faster annealing (was 40)

# Disabled features
use_kan_seq_head=false
kan_grid_size=5
kan_spline_order=3

use_hyperbolic=true # use hyperbolic attention
hyperbolic_curvature=1.0

use_topo_features=false
topo_dim=64
lambda_topo=0.1

# Feature modes
node_feats_mode="1111"
edge_feats_mode="1111"
interface_only=1
mode="111"

# Callbacks
early_stopping_patience=10
checkpoint_mode="min"

# WandB
wandb_enabled=true
wandb_project="antigenforce"

# Run name
run_name="p4-v3-mdn-pairwise-disabled-aux-hyperbolic-cdr1"

# ==============================================================================
# CLI OVERRIDES - These override the above config
# ==============================================================================

usage() {
    echo "Usage: $0 [--gpu <id>] [--epochs <n>]"
    echo ""
    echo "All other parameters are configured in the script itself."
    echo "Edit the CONFIG section at the top of this file."
    exit 1
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

# Parse CLI args (optional overrides)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu) gpu="$2"; shift 2 ;;
        --epochs) max_epoch="$2"; shift 2 ;;
        *) echo "Unknown parameter: $1"; usage ;;
    esac
done

# ==============================================================================
# RUN TRAINING
# ==============================================================================

mkdir -p logs

timestamp=$(date +%Y%m%d-%H%M%S)
log_file="logs/${timestamp}.log"

echo "=========================================="
echo "HierAb Training (code_v1: NO PLM)"
echo "Config from hyperparameter sweep"
echo "=========================================="
echo "GPU: $gpu | Epochs: $max_epoch | Batch: $batch_size | LR: $lr"
echo "Split: $split | CDR: H$cdr_type"
echo "ESM: $use_esm | Pairwise: $use_pairwise | Iterative: $use_iterative"
echo "Log: $log_file"
echo "=========================================="

nohup python -u trainer.py \
    training.gpu=$gpu \
    training.max_epoch=$max_epoch \
    training.batch_size=$batch_size \
    training.lr=$lr \
    training.weight_decay=$weight_decay \
    training.anneal_base=$anneal_base \
    training.grad_clip=$grad_clip \
    training.num_workers=$num_workers \
    training.seed=$seed \
    training.run_name="$run_name" \
    dataset.split=$split \
    dataset.cdr_type=\"$cdr_type\" \
    model.embed_size=$embed_size \
    model.hidden_size=$hidden_size \
    model.n_layers=$n_layers \
    model.dropout=$dropout \
    model.num_attn_heads=$num_attn_heads \
    model.n_virtual_nodes=$n_virtual_nodes \
    model.alpha=$alpha \
    model.beta=$beta \
    model.gamma=$gamma \
    model.delta=$delta \
    model.epsilon=$epsilon \
    model.dock_cutoff=$dock_cutoff \
    model.use_direct_attn=$use_direct_attn \
    model.lambda_antigen_cls=$lambda_antigen_cls \
    model.lambda_infonce=$lambda_infonce \
    model.lambda_mine=$lambda_mine \
    model.zeta=$zeta \
    model.infonce_temp=$infonce_temp \
    model.latent_dim=$latent_dim \
    model.p_fw_dropout=$p_fw_dropout \
    model.eta=$eta \
    model.use_esm=$use_esm \
    model.esm_dim=$esm_dim \
    model.esm_model=$esm_model \
    model.use_learned_queries=$use_learned_queries \
    model.init_gate_bias=$init_gate_bias \
    model.init_bypass_alpha=$init_bypass_alpha \
    model.use_kan_seq_head=$use_kan_seq_head \
    model.kan_grid_size=$kan_grid_size \
    model.kan_spline_order=$kan_spline_order \
    model.use_hyperbolic=$use_hyperbolic \
    model.hyperbolic_curvature=$hyperbolic_curvature \
    model.use_topo_features=$use_topo_features \
    model.topo_dim=$topo_dim \
    model.lambda_topo=$lambda_topo \
    model.use_pairwise=$use_pairwise \
    model.use_iterative=$use_iterative \
    model.n_refine_passes=$n_refine_passes \
    model.n_message_passes=$n_message_passes \
    model.pairwise_coupling_mode=$pairwise_coupling_mode \
    model.lambda_pairwise=$lambda_pairwise \
    model.gumbel_tau=$gumbel_tau \
    model.use_improved_mine=$use_improved_mine \
    model.mine_type=$mine_type \
    model.mine_n_negatives=$mine_n_negatives \
    model.use_mdn=$use_mdn \
    model.mdn_k=$mdn_k \
    model.use_amcl=$use_amcl \
    model.amcl_tau_start=$amcl_tau_start \
    model.amcl_tau_end=$amcl_tau_end \
    model.amcl_anneal_epochs=$amcl_anneal_epochs \
    model.node_feats_mode=\"$node_feats_mode\" \
    model.edge_feats_mode=\"$edge_feats_mode\" \
    model.interface_only=$interface_only \
    model.mode=\"$mode\" \
    callbacks.early_stopping.patience=$early_stopping_patience \
    callbacks.checkpoint.mode=$checkpoint_mode \
    wandb.enabled=$wandb_enabled \
    wandb.project=$wandb_project \
    > "$log_file" 2>&1 &

pid=$!
echo "Started (PID: $pid)"
echo "Monitor: tail -f $log_file"
echo "Kill: kill $pid"
