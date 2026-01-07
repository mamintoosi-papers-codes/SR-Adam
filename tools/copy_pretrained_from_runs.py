"""Copy best checkpoints for specified datasets/optimizers (batch-size) into a pretrained folder.

Usage:
  python tools/copy_pretrained_from_runs.py --datasets CIFAR10 CIFAR100 --noise 0.05 --batch_size 512 --runs-root runs_2025 --out pretrained-models

Behavior:
- For each dataset and optimizer (Adam, SR-Adam):
  - Finds the per-run meta files under results/.../batch_size_{bs}/run_*_meta.json and picks the run with highest `best_test_acc`.
  - Looks for the corresponding `run_{id}_best.pt` in `--runs-root` (tries both with and without batch_size subdir).
  - Copies the checkpoint into the output folder with a descriptive name: `{Optimizer}_{Dataset}_bs{bs}_best.pt`.

Notes:
- This script runs locally and requires Python and access to the repo files.
"""

import argparse
from pathlib import Path
import json
import shutil
import sys

OPTIMIZERS = ['Adam', 'SR-Adam']


def find_best_run_id(results_root: Path, dataset: str, model: str, noise: str, optimizer: str, batch_size: int):
    folder = results_root / dataset / model / f'noise_{noise}' / optimizer / f'batch_size_{batch_size}'
    if not folder.exists():
        return None
    best_id = None
    best_acc = -1.0
    for meta in folder.glob('run_*_meta.json'):
        try:
            with open(meta, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            continue
        acc = data.get('best_test_acc') or data.get('final_test_acc') or -1
        rid = data.get('run_id')
        if rid is None:
            # try to infer id from filename
            name = meta.stem  # run_{id}_meta
            parts = name.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                rid = int(parts[1])
        if rid is None:
            continue
        if acc > best_acc:
            best_acc = acc
            best_id = rid
    return best_id


def find_checkpoint(runs_root: Path, dataset: str, model: str, noise: str, optimizer: str, batch_size: int, run_id: int):
    candidates = []
    # new format: with batch_size subdir
    candidates.append(runs_root / dataset / model / f'noise_{noise}' / optimizer / f'batch_size_{batch_size}' / f'run_{run_id}_best.pt')
    # older/alternate format: without batch_size
    candidates.append(runs_root / dataset / model / f'noise_{noise}' / optimizer / f'run_{run_id}_best.pt')
    # also try runs_root/runs_*/ pattern
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='+', default=['CIFAR10','CIFAR100'])
    p.add_argument('--model', default='simplecnn')
    p.add_argument('--noise', default='0.05')
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--results-root', default='results')
    p.add_argument('--runs-root', default='runs_2025')
    p.add_argument('--out', default='pretrained-models')
    args = p.parse_args()

    results_root = Path(args.results_root)
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for ds in args.datasets:
        for opt in OPTIMIZERS:
            run_id = find_best_run_id(results_root, ds, args.model, args.noise, opt, args.batch_size)
            if run_id is None:
                summary.append({'dataset': ds, 'optimizer': opt, 'status': 'no_runs_meta_found'})
                continue
            ckpt = find_checkpoint(runs_root, ds, args.model, args.noise, opt, args.batch_size, run_id)
            if ckpt is None:
                summary.append({'dataset': ds, 'optimizer': opt, 'status': 'checkpoint_not_found', 'run_id': run_id})
                continue
            dest = out_dir / f"{opt}_{ds}_bs{args.batch_size}_best.pt"
            shutil.copy2(ckpt, dest)
            summary.append({'dataset': ds, 'optimizer': opt, 'status': 'copied', 'src': str(ckpt), 'dst': str(dest)})

    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
