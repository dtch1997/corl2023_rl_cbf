#!/bin/bash

# Baseline
python experiments/evaluation/download_checkpoints.py \
  --group-name cartpole_ablations_seeded_3 \
  --exp-name baseline_2M \
  --save-dir experiments/artifacts/checkpoints

# Bump only
python experiments/evaluation/download_checkpoints.py \
  --group-name cartpole_ablations_seeded_3 \
  --exp-name bump_2M \
  --save-dir experiments/artifacts/checkpoints

# Supervision only
python experiments/evaluation/download_checkpoints.py \
  --group-name cartpole_ablations_seeded_3 \
  --exp-name baseline_supervised_2M \
  --save-dir experiments/artifacts/checkpoints

# Bump, supervised, 
python experiments/evaluation/download_checkpoints.py \
  --group-name cartpole_ablations_seeded_3 \
  --exp-name bump_supervised_2M \
  --save-dir experiments/artifacts/checkpoints

# Bump, supervised, base env
python experiments/evaluation/download_checkpoints.py \
  --group-name cartpole_ablations_seeded_3 \
  --exp-name bump_supervised_base_2M  \
  --save-dir experiments/artifacts/checkpoints