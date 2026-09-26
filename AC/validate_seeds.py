"""Repeat the declared convergence test with independent random seeds."""
import argparse
import json
from pathlib import Path
from config import config
from train import train, save_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3, 42])
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).resolve().parents[2]/'output'/'ac_converged'/'seed_validation')
    args = parser.parse_args()
    summaries = []
    for seed in args.seeds:
        cfg = config(seed=seed)
        runner, history, evaluations, result = train(cfg, verbose=False)
        save_run(args.output/f'seed_{seed}', cfg, runner, history, evaluations, result)
        summary = dict(seed=seed, converged=result['converged'], updates=result['updates'], **result['metrics'])
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    (args.output/'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    if not all(r['converged'] for r in summaries):
        raise RuntimeError('Some seeds failed the convergence criteria; see saved diagnostics')


if __name__ == '__main__':
    main()
