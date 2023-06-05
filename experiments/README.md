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

Here we explain how to reproduce figures from the paper. 

One pre-trained checkpoint per experimental setting is available in `experiments/artifacts/checkpoints`. 

Metrics for 5 seeded runs of each experimental setting is available in `experiments/artifacts/data/metrics.csv`

Collection of pre-generated plots is available in `experiments/artifacts/plots`. Below, we provide instructions for generating all plots. 


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