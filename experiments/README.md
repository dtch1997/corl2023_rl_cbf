# Experiments

Collection of scripts to reproduce experiments presented in the paper. 

Run all scripts from top level directory, e.g.
```
bash experiments/XXX/YYY.sh
```

## Training

Run experiments on CartPole:
```
bash experiments/training/cartpole.sh
```

## Evaluation

Here we explain how to reproducce figures from the paper. 

### Download

Download 1 pre-trained checkpoint per experimental setting
```
bash experiments/evaluation/download_checkpoints.sh
```

Download data for 5 seeds per experimental setting
```
bash experiments/evaluation/download_data.sh
```

### Visualize training history

Calculate and plot metrics over training history
```
bash experiments/visualization/viz_metrics.sh
```

### Visualize learned barrier functions

Plot 1 barrier function per experimental setting
```
bash experiments/visualization/viz_barrier.sh
```

### Generate safety-constrained trajectory
```
bash experiments/evaluation/eval_constrain.sh
```

### Evaluate CBFs as safety constraint
```
bash experiments/evaluation/eval_constrain_random.sh
bash experiments/evaluation/viz_constrain_random.sh
```

### Generate other plots
```
# Tradeoff, barrier coverage vs barrier validity
python experiments/visualization/viz_metric_tradeoff.py \
    -i experiments/artifacts/data/metrics.csv \
    -m1 eval/barrier/validity_alpha_0.9 \
    -m2 eval/barrier/coverage \
    -o experiments/artifacts/plots/viz_barrier_coverage_vs_validity.png
```