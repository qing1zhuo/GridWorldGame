"""Deterministic dynamic-programming baseline for the PPO GridWorld.

The functions in this module do not train the policy.  They enumerate the
finite transition model, solve the Bellman optimality equation with value
iteration, and compare the Actor's deterministic (argmax) policy with that
reference.  Target cells are treated as terminal states with value zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import GridWorld_Config


ACTION_NAMES = ("up", "right", "down", "left", "stay")
ACTION_SYMBOLS = ("^", ">", "v", "<", "o")


def _validated_grid(cfg: GridWorld_Config) -> np.ndarray:
    grid = np.asarray(cfg.grid, dtype=np.int64)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("cfg.grid must be a non-empty rectangular 2D array")
    if grid.shape != (cfg.row_num, cfg.col_num):
        raise ValueError("cfg.grid shape must match row_num and col_num")
    known_cells = set(cfg.idx2grid)
    if not set(np.unique(grid)).issubset(known_cells):
        raise ValueError("cfg.grid contains a cell id missing from cfg.idx2grid")
    if cfg.input_dim != grid.size:
        raise ValueError("cfg.input_dim must equal row_num * col_num")
    if cfg.action_dim != len(ACTION_NAMES):
        raise ValueError("this GridWorld baseline requires five actions")
    if not np.isfinite(cfg.gamma) or not 0.0 <= cfg.gamma < 1.0:
        raise ValueError("cfg.gamma must be finite and satisfy 0 <= gamma < 1")
    required_rewards = {"blank", "boundary", "forbidden", "target"}
    if not required_rewards.issubset(cfg.rewards):
        raise ValueError(f"cfg.rewards must define {sorted(required_rewards)}")
    if not all(np.isfinite(cfg.rewards[name]) for name in required_rewards):
        raise ValueError("all rewards must be finite")
    return grid


def transition_model(cfg: GridWorld_Config):
    """Enumerate next state, reward, and termination for every state/action.

    The returned arrays have shape ``(row_num * col_num, action_dim)``.  A
    boundary action remains in the current state.  Rows belonging to target
    cells are absorbing terminal rows because those cells are never valid
    episode starts in :class:`env.GridWorld`.
    """
    grid = _validated_grid(cfg)
    state_count = grid.size
    action_count = cfg.action_dim
    dx = np.asarray([cfg.action2dx[a] for a in range(action_count)], dtype=np.int64)
    dy = np.asarray([cfg.action2dy[a] for a in range(action_count)], dtype=np.int64)

    next_states = np.empty((state_count, action_count), dtype=np.int64)
    rewards = np.empty((state_count, action_count), dtype=np.float64)
    dones = np.zeros((state_count, action_count), dtype=bool)
    terminal = np.zeros(state_count, dtype=bool)

    for state in range(state_count):
        row, col = divmod(state, cfg.col_num)
        if cfg.idx2grid[int(grid[row, col])] == "target":
            next_states[state] = state
            rewards[state] = 0.0
            dones[state] = True
            terminal[state] = True
            continue

        for action in range(action_count):
            new_row = row + int(dx[action])
            new_col = col + int(dy[action])
            if not (0 <= new_row < cfg.row_num and 0 <= new_col < cfg.col_num):
                next_states[state, action] = state
                rewards[state, action] = cfg.rewards["boundary"]
                continue

            next_state = new_row * cfg.col_num + new_col
            cell_name = cfg.idx2grid[int(grid[new_row, new_col])]
            next_states[state, action] = next_state
            rewards[state, action] = cfg.rewards[cell_name]
            dones[state, action] = cell_name == "target"

    return next_states, rewards, dones, terminal


def optimal_reference(
    cfg: GridWorld_Config,
    tolerance: float = 1e-12,
    max_iterations: int = 100_000,
    action_tolerance: float = 1e-9,
):
    """Compute an optimal deterministic baseline using value iteration.

    ``optimal_actions`` retains all tied optimal actions.  ``policy`` uses the
    lowest action index to make the representative baseline deterministic and
    stores ``-1`` on terminal cells.
    """
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(action_tolerance) or action_tolerance < 0:
        raise ValueError("action_tolerance must be finite and non-negative")

    next_states, rewards, dones, terminal = transition_model(cfg)
    values = np.zeros(next_states.shape[0], dtype=np.float64)
    active = ~terminal

    for iteration in range(1, max_iterations + 1):
        q_values = rewards + cfg.gamma * (~dones) * values[next_states]
        updated = values.copy()
        updated[active] = q_values[active].max(axis=1)
        residual = float(np.max(np.abs(updated[active] - values[active])))
        values = updated
        if residual <= tolerance:
            break
    else:
        raise RuntimeError(
            "value iteration did not converge after "
            f"{max_iterations} iterations (residual={residual:.3g})"
        )

    q_values = rewards + cfg.gamma * (~dones) * values[next_states]
    optimal_actions = np.isclose(
        q_values,
        q_values.max(axis=1, keepdims=True),
        atol=action_tolerance,
        rtol=0.0,
    )
    optimal_actions[terminal] = False
    policy = optimal_actions.argmax(axis=1).astype(np.int64)
    policy[terminal] = -1
    shape = (cfg.row_num, cfg.col_num)

    return {
        "values": values.reshape(shape),
        "q": q_values.reshape(*shape, cfg.action_dim),
        "policy": policy.reshape(shape),
        "optimal_actions": optimal_actions.reshape(*shape, cfg.action_dim),
        "terminal_mask": terminal.reshape(shape),
        "iterations": iteration,
        "bellman_residual": residual,
        "value_error_bound": residual / (1.0 - cfg.gamma),
    }


def evaluate_policy(policy, cfg: GridWorld_Config, reference=None):
    """Exactly evaluate a deterministic policy and compare it with baseline.

    Policy values are obtained from the linear Bellman system, not from a
    finite Monte-Carlo rollout.  Metrics use every nonterminal grid cell.
    """
    policy = np.asarray(policy)
    shape = (cfg.row_num, cfg.col_num)
    if policy.shape == (cfg.row_num * cfg.col_num,):
        policy = policy.reshape(shape)
    if policy.shape != shape:
        raise ValueError(f"policy must have shape {shape}")

    next_states, rewards, dones, terminal = transition_model(cfg)
    actions = policy.reshape(-1)
    active_ids = np.flatnonzero(~terminal)
    if not np.isin(actions[active_ids], np.arange(cfg.action_dim)).all():
        raise ValueError("nonterminal policy entries must be action indices 0..4")

    active_lookup = np.full(actions.size, -1, dtype=np.int64)
    active_lookup[active_ids] = np.arange(active_ids.size)
    matrix = np.eye(active_ids.size, dtype=np.float64)
    rhs = rewards[active_ids, actions[active_ids]].astype(np.float64)

    chosen_next = next_states[active_ids, actions[active_ids]]
    chosen_done = dones[active_ids, actions[active_ids]]
    bootstraps = (~chosen_done) & (~terminal[chosen_next])
    rows = np.flatnonzero(bootstraps)
    matrix[rows, active_lookup[chosen_next[bootstraps]]] -= cfg.gamma
    active_values = np.linalg.solve(matrix, rhs)

    values = np.zeros(actions.size, dtype=np.float64)
    values[active_ids] = active_values
    if reference is None:
        reference = optimal_reference(cfg)
    reference_values = np.asarray(reference["values"], dtype=np.float64).reshape(-1)
    reference_actions = np.asarray(reference["optimal_actions"], dtype=bool).reshape(
        -1, cfg.action_dim
    )
    correct = reference_actions[active_ids, actions[active_ids]]
    value_gap = reference_values - values
    active_gap = value_gap[active_ids]

    correct_grid = np.ones(actions.size, dtype=bool)
    correct_grid[active_ids] = correct
    return {
        "values": values.reshape(shape),
        "value_gap": value_gap.reshape(shape),
        "optimal_action_mask": correct_grid.reshape(shape),
        "metrics": {
            "nonterminal_states": int(active_ids.size),
            "optimal_action_fraction": float(correct.mean()),
            "mean_value_gap": float(active_gap.mean()),
            "max_value_gap": float(active_gap.max()),
            "max_absolute_value_gap": float(np.abs(active_gap).max()),
            "mean_policy_value": float(active_values.mean()),
            "mean_optimal_value": float(reference_values[active_ids].mean()),
        },
    }


def deterministic_actor_policy(runner, cfg: GridWorld_Config):
    """Return Actor argmax actions, probabilities, and Critic values."""
    states = np.arange(cfg.row_num * cfg.col_num, dtype=np.int64)
    state_tensor = torch.as_tensor(states, dtype=torch.long)
    actor_training = runner.actor.training
    critic_training = runner.critic.training
    runner.actor.eval()
    runner.critic.eval()
    try:
        with torch.no_grad():
            logits = runner.actor(state_tensor)
            probabilities = torch.softmax(logits, dim=-1)
            actions = logits.argmax(dim=-1)
            critic_values = runner.critic(state_tensor).squeeze(-1)
    finally:
        runner.actor.train(actor_training)
        runner.critic.train(critic_training)

    shape = (cfg.row_num, cfg.col_num)
    return (
        actions.cpu().numpy().reshape(shape),
        probabilities.cpu().numpy().reshape(*shape, cfg.action_dim),
        critic_values.cpu().numpy().reshape(shape),
    )


def evaluate(runner, cfg: GridWorld_Config, reference=None):
    """Compare the PPO Actor's deterministic deployment policy to baseline."""
    if reference is None:
        reference = optimal_reference(cfg)
    policy, probabilities, critic_values = deterministic_actor_policy(runner, cfg)
    result = evaluate_policy(policy, cfg, reference)
    result.update(
        policy=policy,
        probabilities=probabilities,
        critic_values=critic_values,
        optimal_values=reference["values"],
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, help="override the default discount factor")
    parser.add_argument("--save", type=Path, help="save baseline arrays to an NPZ file")
    args = parser.parse_args()

    cfg = GridWorld_Config()
    if args.gamma is not None:
        cfg.gamma = args.gamma
    reference = optimal_reference(cfg)

    print("Optimal deterministic baseline (^ > v < o; ties are retained):")
    for row, terminal_row in zip(reference["optimal_actions"], reference["terminal_mask"]):
        cells = []
        for mask, is_terminal in zip(row, terminal_row):
            cells.append("T" if is_terminal else "".join(
                symbol for symbol, keep in zip(ACTION_SYMBOLS, mask) if keep
            ))
        print("  ".join(f"{cell:^5}" for cell in cells))
    print("Optimal state values (terminal cells are zero):")
    print(np.array2string(reference["values"], precision=6, suppress_small=True))
    print(
        f"iterations={reference['iterations']}, "
        f"Bellman residual={reference['bellman_residual']:.3e}"
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        with args.save.open("wb") as output:
            np.savez_compressed(
                output,
                **reference,
                grid=np.asarray(cfg.grid),
                gamma=cfg.gamma,
                action_names=np.asarray(ACTION_NAMES),
            )
        print(f"Saved baseline to {args.save}")


if __name__ == "__main__":
    main()
