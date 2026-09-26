"""Exact finite-MDP evaluation. Used ONLY for diagnostics and stopping criteria."""
import numpy as np
import torch
from env import GridWorld


def transition_model(cfg):
    return GridWorld(cfg).transition_batch(np.arange(cfg.net.input_dim)[:, None], np.arange(5)[None, :])


def optimal_reference(cfg):
    nxt, rewards, done = transition_model(cfg)
    values = np.zeros(cfg.net.input_dim)
    for _ in range(cfg.rl.value_iteration_max_steps):
        q = rewards + cfg.rl.gamma * (~done) * values[nxt]
        new_values = q.max(axis=1)
        if np.max(np.abs(new_values-values)) < cfg.rl.value_iteration_tolerance:
            values = new_values
            break
        values = new_values
    else:
        raise RuntimeError("Value iteration failed to converge")
    q = rewards + cfg.rl.gamma * (~done) * values[nxt]
    return dict(values=values, q=q, optimal_actions=np.isclose(
        q, values[:, None], atol=cfg.rl.optimal_action_tolerance, rtol=0))


def policy_values(probabilities, cfg):
    """Solve (I - gamma P_pi) V_pi = r_pi, including zero-reward loops."""
    nxt, rewards, done = transition_model(cfg)
    count = cfg.net.input_dim
    p = np.zeros((count, count))
    for s in range(count):
        np.add.at(p[s], nxt[s], probabilities[s] * (~done[s]))
    r = (probabilities * rewards).sum(axis=1)
    return np.linalg.solve(np.eye(count) - cfg.rl.gamma * p, r)


def evaluate(runner, cfg, reference=None):
    if reference is None:
        reference = optimal_reference(cfg)
    with torch.no_grad():
        features = torch.eye(cfg.net.input_dim, device=runner.device)
        logits, critic = runner.model(features)
        probs = logits.softmax(-1).cpu().numpy().astype(np.float64)
        probs /= probs.sum(axis=1, keepdims=True)
        critic = critic.cpu().numpy().astype(np.float64)
    policy = probs.argmax(axis=1)
    v_pi = policy_values(probs, cfg)
    v_greedy = policy_values(np.eye(5)[policy], cfg)
    mask = cfg.env.grid.ravel() != 2
    ids = np.flatnonzero(mask)
    correct = reference['optimal_actions'][ids, policy[ids]]
    metrics = dict(
        optimal_action_fraction=float(correct.mean()),
        greedy_max_gap=float(np.max(np.abs(reference['values'][mask]-v_greedy[mask]))),
        stochastic_max_gap=float(np.max(np.abs(reference['values'][mask]-v_pi[mask]))),
        critic_max_error=float(np.max(np.abs(critic[mask]-v_pi[mask]))),
        greedy_mean_value=float(v_greedy[mask].mean()),
        stochastic_mean_value=float(v_pi[mask].mean()),
        optimal_mean_value=float(reference['values'][mask].mean()),
        min_optimal_action_mass=float((probs * reference['optimal_actions']).sum(1)[mask].min()))
    passed = (metrics['optimal_action_fraction'] == 1.0
              and metrics['greedy_max_gap'] < cfg.rl.greedy_tolerance
              and metrics['stochastic_max_gap'] <= cfg.rl.policy_tolerance
              and metrics['critic_max_error'] <= cfg.rl.critic_tolerance)
    return dict(metrics=metrics, passed=passed, policy=policy, probabilities=probs,
                critic=critic, stochastic_values=v_pi, greedy_values=v_greedy,
                optimal_values=reference['values'])
