#!/bin/bash

python experiments/visualization/viz_constrain_random.py \
    --model-path experiments/artifacts/checkpoints/bump_supervised_2M.pth \
    --data-path experiments/artifacts/data/eval_constrain_random/bump_supervised_2M_states.csv \
    --save-path experiments/artifacts/plots/viz_eval_constrain.png \
    --device cpu