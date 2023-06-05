import argparse
import wandb
import pathlib

ENTITY = 'dtch1997'
PROJECT = 'RL_CBF'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', type=str, default=ENTITY)
    parser.add_argument('--project', type=str, default=PROJECT)
    parser.add_argument('--exp-name', type=str, required=True)
    parser.add_argument('--group-name', type=str, default = None)
    parser.add_argument('--seed', type=int, default = None)
    parser.add_argument('--save-dir', type=str, default = None)
    return parser.parse_args()

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
    for i, run in enumerate(runs):
        print(i, run.id)
        wandb_filename = args.exp_name + '.pth'
        checkpoint = run.file(wandb_filename)
        seed = run.config['seed']

        print(checkpoint.name)

        if args.save_dir:
            save_path = pathlib.Path(args.save_dir) / f'{args.exp_name}_{seed}.pth'
            checkpoint.download(".", replace=True)
            download_fp = pathlib.Path(wandb_filename)
            print("Downloaded to: ", download_fp)
            download_fp.rename(save_path)
            print("Moved to: ", save_path)

        