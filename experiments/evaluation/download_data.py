import argparse
import wandb
import pathlib
import pandas as pd
import concurrent.futures

ENTITY = 'dtch1997'
PROJECT = 'RL_CBF'

METRICS=[
    'global_step',
    'eval/rollout/episode_length',
    'eval/rollout/episode_return',
    'eval/grid/mean_td_error',
    'eval/barrier/coverage',
    # Note, 'alpha' here is actually one minus alpha
    'eval/barrier/validity_alpha_0.9',
    'eval/constrain/mean_episode_length'
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=str, default=ENTITY)
    parser.add_argument('--project', type=str, default=PROJECT)
    parser.add_argument('--group-name', type=str, default = None)
    parser.add_argument('--exp-name', type=str, default=None)
    parser.add_argument('--seed', type=int, default = None)
    parser.add_argument('--save-dir', type=str, default = None)
    parser.add_argument('--num-samples', type=int, default=200,
                        help="Number of samples.")
    return parser.parse_args()

def download_metrics(run, num_samples: int = None) -> pd.DataFrame:
    df = run.history(samples=num_samples, keys=METRICS, pandas=True)
    df['run_id'] = run.id
    df['exp_name'] = run.config['exp_name']
    df['seed'] = run.config['seed']
    return df

if __name__ == "__main__":

    args = get_args()

    api = wandb.Api()

    filters = {}
    if args.exp_name is not None:
        filters['config.exp_name'] = args.exp_name
    if args.group_name is not None:
        filters['group'] = args.group_name
    if args.seed is not None:
        filters['config.seed'] = args.seed

    runs = api.runs(f"{ENTITY}/{PROJECT}", filters = filters)
    rows = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(download_metrics, run, args.num_samples) for run in runs]
    results = [future.result() for future in futures]        
    df = pd.concat(results)
    df.to_csv(pathlib.Path(args.save_dir) / 'metrics.csv')    