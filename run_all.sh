#!/bin/bash
# REASAN Training Pipeline
# This script runs the complete training pipeline for all three policies:
# 1. Locomotion policy (two stages)
# 2. Safety shield/filter policy (two stages)
# 3. Navigation policy (single stage)

set -e  # Exit on error

# Change to training directory (required for scripts to work)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="${SCRIPT_DIR}/training"
cd "${TRAINING_DIR}"

echo "=========================================="
echo "Starting REASAN Training Pipeline"
echo "=========================================="

# Stage 1: Locomotion Policy - First Stage
echo ""
echo "=========================================="
echo "Training Locomotion Policy - Stage 1"
echo "=========================================="
python scripts/train_loco.py --run_name loco_new --num_envs 8 --max_iterations 5000 --wandb_proj go2_loco

# Stage 2: Locomotion Policy - Second Stage (higher speed and robustness)
echo ""
echo "=========================================="
echo "Training Locomotion Policy - Stage 2"
echo "=========================================="
python scripts/train_loco.py --run_name loco_new --resume --load_run loco_new --num_envs 8 --max_iterations 5000 --wandb_proj go2_loco --second_stage

# Stage 3: Safety Shield/Filter Policy - First Stage
echo ""
echo "=========================================="
echo "Training Safety Shield Policy - Stage 1"
echo "=========================================="
python scripts/train_filter.py --run_name filter_new --num_envs 8 --max_iterations 10000 --wandb_proj go2_filter --confirm

# Stage 4: Safety Shield/Filter Policy - Second Stage (with dynamic obstacles)
echo ""
echo "=========================================="
echo "Training Safety Shield Policy - Stage 2 (with dynamic obstacles)"
echo "=========================================="
python scripts/train_filter.py --run_name filter_new --resume --load_run filter_new --num_envs 8 --max_iterations 10000 --wandb_proj go2_filter --with_dyn_obst --confirm

# Stage 5: Navigation Policy (single stage with dynamic obstacles)
echo ""
echo "=========================================="
echo "Training Navigation Policy"
echo "=========================================="
python scripts/train_nav.py --run_name nav_new --num_envs 8 --max_iterations 10000 --wandb_proj go2_nav --with_dyn_obst --confirm

echo ""
echo "=========================================="
echo "Training Pipeline Complete!"
echo "=========================================="
