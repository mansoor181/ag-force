#!/bin/bash
# Launch AntigenForce (P4) v4 sweep agents on multiple GPUs in parallel.
#
# Default: 20 total trials (10 per GPU on 2 GPUs)
#
# Usage:
#   bash launch_sweep.sh                    # GPUs 0,1; 10 trials each = 20 total
#   bash launch_sweep.sh --gpus 0,1 --count 15
#   bash launch_sweep.sh --sweep_id abc123  # Join existing sweep

set -euo pipefail
cd "$(dirname "$0")"

# Defaults
GPUS="0,1"
COUNT=10  # Per agent (2 agents = 20 total)
SWEEP_ID=""
PYTHON="python"  # Update to your Python path if needed

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)     GPUS="$2"; shift 2 ;;
        --count)    COUNT="$2"; shift 2 ;;
        --sweep_id) SWEEP_ID="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p logs

# Create sweep if not provided
if [ -z "$SWEEP_ID" ]; then
    SWEEP_ID=$($PYTHON -c "
import wandb, yaml
with open('sweep.yaml') as f:
    cfg = yaml.safe_load(f)
sid = wandb.sweep(cfg, project='antigenforce')
print(sid)
" 2>/dev/null | tail -1)
    echo "Created sweep: $SWEEP_ID"
    echo "URL: https://wandb.ai/<your-entity>/antigenforce/sweeps/$SWEEP_ID"
fi

# Launch one agent per GPU
IFS=',' read -ra GPU_LIST <<< "$GPUS"
PIDS=()

for gpu in "${GPU_LIST[@]}"; do
    LOG="logs/sweep_gpu${gpu}.log"
    echo "Launching agent on GPU $gpu ($COUNT trials) -> $LOG"
    CUDA_VISIBLE_DEVICES="$gpu" nohup $PYTHON run_sweep.py \
        --gpu 0 \
        --sweep_id "$SWEEP_ID" \
        --count "$COUNT" \
        > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "  PID: ${PIDS[-1]}"
done

N_AGENTS=${#PIDS[@]}
TOTAL_TRIALS=$(($N_AGENTS * COUNT))

echo ""
echo "============================================"
echo "Sweep ID: $SWEEP_ID"
echo "GPUs: ${GPU_LIST[*]}"
echo "Trials per agent: $COUNT"
echo "Total trials: $TOTAL_TRIALS"
echo "PIDs: ${PIDS[*]}"
echo "============================================"
echo ""
echo "Monitor:"
echo "  tail -f logs/sweep_gpu*.log"
echo "  wandb: https://wandb.ai/<your-entity>/antigenforce/sweeps/$SWEEP_ID"
echo ""
echo "Metric: val_caar (maximize) - targeting >0.25"
