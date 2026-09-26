"""Finite-grid sampled Actor-Critic configuration."""
from types import SimpleNamespace
import numpy as np

class config:
    def __init__(self, row_num=5, col_num=5, reward=None, grid=None,
                 gamma=0.9, lr=0.03, critic_lr=0.03, batch_size=32,
                 train_iteration=12000, device="cpu", entropy_coef=0.02,
                 entropy_decay_steps=2000, max_grad_norm=1.0,
                 eval_interval=100, patience=5, policy_tolerance=0.05,
                 critic_tolerance=0.1, seed=123456,
                 lr_decay_steps=6000, min_lr_ratio=0.1, reward_scale=None,
                 greedy_tolerance=1e-8, value_iteration_tolerance=1e-12,
                 value_iteration_max_steps=10000, optimal_action_tolerance=1e-9,
                 num_threads=1, deterministic_algorithms=True):
        if grid is None:
            grid = [[0,0,0,0,1], [0,1,1,0,2], [1,0,1,0,0], [0,1,2,1,0], [0,1,0,0,0]]
        if reward is None:
            reward = dict(boundary=-10, forbidden=-10, target=10, blank=0)
        grid = np.asarray(grid, dtype=np.int64)
        if grid.shape != (row_num, col_num) or not np.isin(grid, [0,1,2]).all():
            raise ValueError("invalid grid shape or cell type")
        if not np.any(grid == 2) or not np.any(grid != 2):
            raise ValueError("need both target and nonterminal states")
        if not 0 <= gamma < 1:
            raise ValueError("require 0 <= gamma < 1")
        if min(batch_size,train_iteration,eval_interval,patience,entropy_decay_steps,
               lr_decay_steps,value_iteration_max_steps,num_threads) < 1:
            raise ValueError("counts and intervals must be positive")
        if not 0 < min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in (0, 1]")
        if any(not np.isfinite(v) or v <= 0 for v in
               (lr, critic_lr, max_grad_norm, greedy_tolerance, value_iteration_tolerance)):
            raise ValueError("learning rates, gradient limit and solver tolerances must be positive and finite")
        if any(not np.isfinite(v) or v < 0 for v in
               (entropy_coef, policy_tolerance, critic_tolerance, optimal_action_tolerance)):
            raise ValueError("entropy coefficient and error tolerances must be nonnegative and finite")
        self.env = SimpleNamespace(row_num=row_num, col_num=col_num, reward=dict(reward), grid=grid.copy())
        self.rl = SimpleNamespace(gamma=gamma, lr=lr, critic_lr=critic_lr, batch_size=batch_size,
            train_iteration=train_iteration, device=device, entropy_coef=entropy_coef,
            entropy_decay_steps=entropy_decay_steps, max_grad_norm=max_grad_norm,
            eval_interval=eval_interval, patience=patience, policy_tolerance=policy_tolerance,
            critic_tolerance=critic_tolerance, lr_decay_steps=lr_decay_steps,
            min_lr_ratio=min_lr_ratio, greedy_tolerance=greedy_tolerance,
            value_iteration_tolerance=value_iteration_tolerance,
            value_iteration_max_steps=value_iteration_max_steps,
            optimal_action_tolerance=optimal_action_tolerance,
            num_threads=num_threads, deterministic_algorithms=deterministic_algorithms)
        self.rand = SimpleNamespace(seed=seed)
        self.net = SimpleNamespace(input_dim=row_num*col_num, action_dim=5)
        self.reward_scale = (max(1.0, *(abs(float(r)) for r in reward.values()))
                             if reward_scale is None else float(reward_scale))
        if not np.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be positive and finite, or None for automatic scaling")

    def to_dict(self):
        return dict(env={**vars(self.env), "grid": self.env.grid.tolist()},
                    rl=vars(self.rl).copy(), rand=vars(self.rand).copy(),
                    net=vars(self.net).copy(), reward_scale=self.reward_scale)
