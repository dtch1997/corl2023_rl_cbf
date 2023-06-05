
import numpy as np

import torch 
import torch.nn as nn

from typing import List
from siren_pytorch import Siren

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):

    def __init__(self, env, 
                 hidden_dim_1: int = 120,
                 hidden_dim_2: int = 84,
                 enable_bump_parametrization: bool = False,
                 enable_siren_layer: bool = False,
                 min_value: float = 0,
                 max_value: float = 100,
                 device: str = 'cuda'):
        super().__init__()
        self.enable_bump_parametrization = enable_bump_parametrization
        # Only used in case of bump parametrization
        self.max = max_value
        # Not implemented yet
        self.min = min_value
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.device = device

        if enable_siren_layer:
            self.network = nn.Sequential(
                Siren(np.array(env.single_observation_space.shape).prod(), hidden_dim_1, c=6, w0=30.),
                nn.Linear(120, 84),
                nn.ELU(),
                nn.Linear(84, env.single_action_space.n),
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
                nn.ELU(),
                nn.Linear(120, 84),
                nn.ELU(),
                nn.Linear(84, env.single_action_space.n),
            )

        if enable_bump_parametrization:
            self.w = lambda x: (self.max - self.min) * torch.sigmoid(x) + self.min
            self.w_inv = lambda x: torch.log((x - self.min) / (self.max - x))
        else:
            self.w = lambda x: x
            self.w_inv = lambda x: x

    @staticmethod
    def add_argparse_args(parser: 'argparse.ArgumentParser'):
        parser.add_argument('--enable-bump-parametrization', action='store_true')
        parser.add_argument('--enable-siren-layer', action='store_true')
        parser.add_argument('--min-value', type=float, default=0)
        parser.add_argument('--max-value', type=float, default=100)
        parser.add_argument('--hidden-dim-1', type=int, default=120)
        parser.add_argument('--hidden-dim-2', type=int, default=84)
        parser.add_argument('--device', type=str, default='cuda')
        return parser
    
    @staticmethod
    def from_argparse_args(env, args, **kwargs):
        return QNetwork(
            env, 
            enable_bump_parametrization=args.enable_bump_parametrization, 
            enable_siren_layer=args.enable_siren_layer,
            hidden_dim_1=args.hidden_dim_1,
            hidden_dim_2=args.hidden_dim_2,
            min_value=args.min_value,
            max_value=args.max_value,
            device=args.device,
            **kwargs
        )
    
    @property
    def barrier_threshold(self):
        return (self.max - self.min) / 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(self.network(x))

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def get_values(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q_values(x).max(dim=-1)[0]
    
    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.get_q_values(x).max(dim=-1)[1]
    
    def get_safety(self, x: torch.Tensor, barrier_threshold: float = None) -> torch.Tensor:
        """ Predict safety according to barrier threshold """
        if barrier_threshold is None:
            barrier_threshold = self.barrier_threshold
        values = self.get_values(x)
        return self.w_inv(values - barrier_threshold) >= 0

    def get_next_safety(self, 
                        x: torch.Tensor, 
                        u: torch.Tensor, 
                        barrier_threshold: float = None) -> torch.Tensor:
        """ Predict safety of next state according to barrier threshold """
        if barrier_threshold is None:
            barrier_threshold = self.barrier_threshold
        q_values = self.get_q_values(x)
        next_values = q_values.gather(1, u.unsqueeze(-1)).squeeze(-1)
        return self.w_inv(next_values - barrier_threshold) >= 0