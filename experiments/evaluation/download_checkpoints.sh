#!/bin/bash

# Baseline
python scripts/download_wandb_checkpoint.py \
  --wandb-run-path dtch1997/RL_CBF/1pdvqzs4 \
  --wandb-filename baseline_2M.pth \
  --save-path experiments/artifacts/checkpoints/baseline_2M.pth

# Supervision only
python scripts/download_wandb_checkpoint.py \
  --wandb-run-path dtch1997/RL_CBF/dl2whuq7 \
  --wandb-filename baseline_supervised_2M.pth \
  --save-path experiments/artifacts/checkpoints/baseline_supervised_2M.pth

# Bump only
python scripts/download_wandb_checkpoint.py \
  --wandb-run-path dtch1997/RL_CBF/ig65dma7 \
  --wandb-filename bump_2M.pth \
  --save-path experiments/artifacts/checkpoints/bump_2M.pth

# Bump, supervised
python scripts/download_wandb_checkpoint.py \
  --wandb-run-path dtch1997/RL_CBF/9q982z28 \
  --wandb-filename bump_supervised_2M.pth \
  --save-path experiments/artifacts/checkpoints/bump_supervised_2M.pth

# Bump, supervised, base env
# TODO: Update wandb filename to reflect base env
python scripts/download_wandb_checkpoint.py \
  --wandb-run-path dtch1997/RL_CBF/nwf5gh2v \
  --wandb-filename bump_supervised_2M.pth \
  --save-path experiments/artifacts/checkpoints/bump_supervised_base_2M.pth