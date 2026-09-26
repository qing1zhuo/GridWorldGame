"""Run: python code/AC/train.py --seed 123456 --output output/ac_converged"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from config import config
from env import GridWorld
from runner import ActorCriticRunner
from evaluate import optimal_reference, evaluate


def seed_everything(seed, num_threads=1, deterministic_algorithms=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)
    torch.use_deterministic_algorithms(deterministic_algorithms)


def train(cfg, verbose=True):
    seed_everything(cfg.rand.seed, cfg.rl.num_threads, cfg.rl.deterministic_algorithms)
    env = GridWorld(cfg)
    runner = ActorCriticRunner(cfg)
    # Exploring starts: fresh on-policy actions from every nonterminal state.
    # An indefinitely long episode therefore cannot monopolize the training data.
    ids = np.repeat(env.nonterminal_ids, cfg.rl.batch_size)
    eye = torch.eye(cfg.net.input_dim, device=runner.device)
    features = eye[torch.as_tensor(ids, device=runner.device)]
    reference = optimal_reference(cfg)  # evaluator only; never passed to train_batch
    history, evaluations = [], []
    consecutive = 0
    initial = evaluate(runner, cfg, reference)
    evaluations.append(dict(iteration=0, **initial['metrics'], passed=initial['passed']))
    started = time.perf_counter()
    for iteration in range(1, cfg.rl.train_iteration + 1):
        with torch.no_grad():
            logits, _ = runner.model(features)
            actions = torch.distributions.Categorical(logits=logits).sample()
        next_ids, rewards, dones = env.transition_batch(ids, actions.cpu().numpy())
        metrics = runner.train_batch(
            features, actions,
            torch.as_tensor(rewards, dtype=torch.float32, device=runner.device),
            eye[torch.as_tensor(next_ids, device=runner.device)],
            torch.as_tensor(dones, dtype=torch.bool, device=runner.device))
        history.append(dict(iteration=iteration, **metrics))
        if iteration % cfg.rl.eval_interval == 0 or iteration == cfg.rl.train_iteration:
            result = evaluate(runner, cfg, reference)
            m = result['metrics']
            consecutive = consecutive + 1 if result['passed'] else 0
            evaluations.append(dict(iteration=iteration, **m, passed=result['passed']))
            if verbose:
                print(f"update={iteration:5d} optimal={m['optimal_action_fraction']:.1%} "
                      f"greedy_gap={m['greedy_max_gap']:.6f} "
                      f"sampled_policy_gap={m['stochastic_max_gap']:.5f} "
                      f"critic_error={m['critic_max_error']:.5f} "
                      f"stable={consecutive}/{cfg.rl.patience}", flush=True)
            if consecutive >= cfg.rl.patience:
                break
    result['converged'] = consecutive >= cfg.rl.patience
    result['updates'] = iteration
    result['environment_samples'] = iteration * len(ids)
    result['elapsed_seconds'] = time.perf_counter() - started
    return runner, history, evaluations, result


def save_run(directory, cfg, runner, history, evaluations, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, records in [('training_history.csv', history), ('evaluation_history.csv', evaluations)]:
        with (directory/name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    summary = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    summary['config'] = cfg.to_dict()
    summary['versions'] = dict(torch=torch.__version__, numpy=np.__version__)
    (directory/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    np.savez(directory/'policy_values.npz', **{k: v for k, v in result.items() if isinstance(v, np.ndarray)})
    states = []
    for sid in range(cfg.net.input_dim):
        row, col = divmod(sid, cfg.env.col_num)
        states.append(dict(row=row+1, column=col+1, cell_type=int(cfg.env.grid[row,col]),
            greedy_action=int(result['policy'][sid]) if cfg.env.grid[row,col] != 2 else 'terminal',
            optimal_value=float(result['optimal_values'][sid]),
            greedy_value=float(result['greedy_values'][sid]),
            stochastic_value=float(result['stochastic_values'][sid]),
            critic_value=float(result['critic'][sid]),
            **{f'p_action_{a}':float(result['probabilities'][sid,a]) for a in range(5)}))
    with (directory/'state_diagnostics.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=states[0].keys())
        writer.writeheader()
        writer.writerows(states)
    torch.save(dict(model=runner.model.state_dict(), config=cfg.to_dict(),
                    updates=runner.updates, converged=result['converged']), directory/'model.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'output'/'ac_converged')
    parser.add_argument('--max-updates', type=int, default=12000)
    args = parser.parse_args()
    cfg = config(seed=args.seed, train_iteration=args.max_updates)
    runner, history, evaluations, result = train(cfg)
    save_run(args.output, cfg, runner, history, evaluations, result)
    if not result['converged']:
        raise RuntimeError("Convergence criteria not met; diagnostics saved. This run is NOT a success.")
    print(json.dumps(result['metrics'], indent=2))


if __name__ == '__main__':
    main()
