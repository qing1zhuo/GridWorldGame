"""Model-based baseline for the DQN GridWorld (no training required).

Run with the default config or a JSON file containing a rectangular 0/1/2 grid::

    python code/DQN/evaluate.py
    python code/DQN/evaluate.py --map map.json --save baseline.npz

In the training notebook, use the SAME cfg as the runner::

    from evaluate import optimal_reference, evaluate_policy
    from visualize import compute_greedy_policy
    reference = optimal_reference(cfg)
    policy, _ = compute_greedy_policy(runner, cfg)
    comparison = evaluate_policy(policy, cfg, reference)
    print(comparison["metrics"])

Actions: 0=up, 1=right, 2=down, 3=left, 4=stay. Forbidden cells are
enterable with a penalty; targets are nonterminal, exactly as in env.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from config import config
from env import GridWorld


ACTION_NAMES = ("up", "right", "down", "left", "stay")
ACTION_SYMBOLS = ("^", ">", "v", "<", "o")


def transition_model(cfg):
    """Enumerate env.step for all cells and actions, including forbidden cells."""
    grid = np.asarray(cfg.env.grid)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("grid must be a non-empty rectangular 2D array")
    if grid.shape != (cfg.env.row_num, cfg.env.col_num):
        raise ValueError("grid shape must match row_num and col_num")
    if not np.isin(grid, (0, 1, 2)).all():
        raise ValueError("grid cells must be 0 (blank), 1 (forbidden), or 2 (target)")
    if cfg.net.output_dim != len(ACTION_NAMES):
        raise ValueError("GridWorld requires output_dim=5")
    if not np.isfinite(cfg.rl.gamma) or not 0 <= cfg.rl.gamma < 1:
        raise ValueError("The continuing-task baseline requires 0 <= gamma < 1")

    env = GridWorld(cfg)
    next_states = np.empty((grid.size, 5), dtype=np.int64)
    rewards = np.empty((grid.size, 5), dtype=np.float64)
    dones = np.empty((grid.size, 5), dtype=bool)
    for state_id in range(grid.size):
        state = divmod(state_id, cfg.env.col_num)
        for action in range(5):
            env.state = state
            (row, col), reward, done = env.step(action)
            next_states[state_id, action] = row * cfg.env.col_num + col
            rewards[state_id, action] = reward
            dones[state_id, action] = done
    if not np.isfinite(rewards).all():
        raise ValueError("All rewards must be finite")
    return next_states, rewards, dones


def optimal_reference(cfg, tolerance=1e-12, max_iterations=100000,
                      action_tolerance=1e-9):
    """Solve the discounted Bellman optimality equation by value iteration.

    Returns values/policy shaped (rows, cols), and q/optimal_actions shaped
    (rows, cols, 5). policy chooses the lowest action index among numerical
    ties; optimal_actions retains ALL ties within action_tolerance. The
    residual and value_error_bound describe numerical accuracy, not learning.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(action_tolerance) or action_tolerance < 0:
        raise ValueError("action_tolerance must be finite and nonnegative")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    nxt, rewards, done = transition_model(cfg)
    gamma = cfg.rl.gamma
    values = np.zeros(nxt.shape[0], dtype=np.float64)
    for iteration in range(1, max_iterations + 1):
        q = rewards + gamma * (~done) * values[nxt]
        updated = q.max(axis=1)
        residual = float(np.max(np.abs(updated - values)))
        if residual <= tolerance:
            break
        values = updated
    else:
        raise RuntimeError(
            f"Value iteration did not converge after {max_iterations} iterations "
            f"(last residual={residual:.3g}); increase max_iterations."
        )

    # q and the residual refer to the same returned value vector.
    optimal_actions = np.isclose(q, q.max(axis=1, keepdims=True),
                                 atol=action_tolerance, rtol=0)
    shape = (cfg.env.row_num, cfg.env.col_num)
    return dict(
        values=values.reshape(shape),
        q=q.reshape(*shape, 5),
        policy=optimal_actions.argmax(axis=1).reshape(shape),
        optimal_actions=optimal_actions.reshape(*shape, 5),
        iterations=iteration,
        bellman_residual=residual,
        value_error_bound=residual / (1 - gamma),
    )


def evaluate_policy(policy, cfg, reference=None):
    """Compare a deterministic policy against the baseline on EVERY cell.

    Pass compute_greedy_policy(runner, cfg)[0] to compare DQN. An optional
    cached reference must have been computed with the same cfg. Policy value
    is solved exactly by linear algebra (up to floating-point error).
    """
    policy = np.asarray(policy)
    shape = (cfg.env.row_num, cfg.env.col_num)
    if policy.shape != shape or not np.isin(policy, np.arange(5)).all():
        raise ValueError("policy must have grid shape and action indices 0..4")
    if reference is None:
        reference = optimal_reference(cfg)
    nxt, rewards, done = transition_model(cfg)
    actions = policy.astype(np.int64).ravel()
    ids = np.arange(actions.size)
    matrix = np.eye(actions.size)
    matrix[ids, nxt[ids, actions]] -= cfg.rl.gamma * (~done[ids, actions])
    values = np.linalg.solve(matrix, rewards[ids, actions]).reshape(shape)
    correct = reference["optimal_actions"].reshape(-1, 5)[ids, actions]
    value_gap = reference["values"] - values
    return dict(
        values=values,
        value_gap=value_gap,
        optimal_action_mask=correct.reshape(shape),
        metrics=dict(
            optimal_action_fraction=float(correct.mean()),
            mean_value_gap=float(value_gap.mean()),
            max_value_gap=float(value_gap.max()),
            mean_policy_value=float(values.mean()),
            mean_optimal_value=float(reference["values"].mean()),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, help="JSON file containing a 2D 0/1/2 array")
    parser.add_argument("--gamma", type=float, help="Override config.py discount factor")
    parser.add_argument("--save", type=Path, help="Save baseline arrays as an NPZ file")
    args = parser.parse_args()
    cfg = config()
    if args.map is not None:
        cfg.env.grid = np.asarray(json.loads(args.map.read_text(encoding="utf-8-sig")))
        if cfg.env.grid.ndim != 2:
            parser.error("--map must contain a rectangular 2D array")
        cfg.env.row_num, cfg.env.col_num = cfg.env.grid.shape
    if args.gamma is not None:
        cfg.rl.gamma = args.gamma

    reference = optimal_reference(cfg)
    print("Grid: 0=blank, 1=forbidden (enterable), 2=target (nonterminal)")
    print(cfg.env.grid)
    print("Optimal directions: ^=up, >=right, v=down, <=left, o=stay")
    print("Multiple symbols in a cell mean equally optimal actions.")
    cells = [["".join(symbol for symbol, optimal in zip(ACTION_SYMBOLS, mask) if optimal)
              for mask in row] for row in reference["optimal_actions"]]
    width = max(len(cell) for row in cells for cell in row) + 2
    for row in cells:
        print("".join(cell.center(width) for cell in row))
    print("Optimal state values:")
    print(np.array2string(reference["values"], precision=6, suppress_small=True))
    print(f"Iterations: {reference['iterations']}; "
          f"Bellman residual: {reference['bellman_residual']:.3g}")
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        # Use a handle so NumPy does not silently append another extension.
        with args.save.open("wb") as output:
            np.savez_compressed(output, **reference, grid=cfg.env.grid,
                                gamma=cfg.rl.gamma, action_names=ACTION_NAMES,
                                reward_names=list(cfg.env.reward),
                                reward_values=list(cfg.env.reward.values()))
        print(f"Saved baseline to {args.save}")


if __name__ == "__main__":
    main()
