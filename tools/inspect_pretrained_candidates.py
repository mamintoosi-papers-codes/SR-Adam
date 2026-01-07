import torch
from pathlib import Path
import json

base = Path('runs_2025')
datasets = ['CIFAR10','CIFAR100']
noise = '0.05'
results = []
for ds in datasets:
    for opt in ['Adam','SR-Adam']:
        folder = base / ds / 'simplecnn' / f'noise_{noise}' / opt
        if not folder.exists():
            continue
        for f in sorted(folder.glob('run_*_best.pt')):
            try:
                ck = torch.load(f, map_location='cpu')
            except Exception as e:
                results.append({'path': str(f), 'error': str(e)})
                continue
            entry = {'path': str(f), 'dataset': ds, 'optimizer': opt}
            # common keys
            for k in ['batch_size','test_acc','best_test_acc','run_id','epoch']:
                if k in ck:
                    entry[k] = ck.get(k)
            # nested meta
            if 'meta' in ck and isinstance(ck['meta'], dict):
                for k in ['batch_size','run_id']:
                    if k in ck['meta']:
                        entry[k] = ck['meta'][k]
            results.append(entry)

print(json.dumps(results, indent=2, ensure_ascii=False))
