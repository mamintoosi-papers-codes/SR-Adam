import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), 'results')

if not os.path.isdir(ROOT):
    sys.exit('results folder not found: {}'.format(ROOT))

for dirpath, dirnames, filenames in os.walk(ROOT):
    for run_idx in range(1, 6):
        prefix = f'run_{run_idx}'

        csv_candidates = [
            f for f in filenames
            if f.startswith(prefix) and f.endswith('.csv') and '_meta.' not in f
        ]
        if csv_candidates:
            csv_candidates.sort(
                key=lambda f: os.path.getmtime(os.path.join(dirpath, f)),
                reverse=True,
            )
            latest = csv_candidates[0]
            for old in csv_candidates[1:]:
                os.remove(os.path.join(dirpath, old))
            target = os.path.join(dirpath, f'{prefix}.csv')
            if latest != f'{prefix}.csv':
                os.replace(os.path.join(dirpath, latest), target)

        meta_candidates = [
            f for f in filenames
            if f.startswith(prefix) and f.endswith('_meta.json')
        ]
        if meta_candidates:
            meta_candidates.sort(
                key=lambda f: os.path.getmtime(os.path.join(dirpath, f)),
                reverse=True,
            )
            latest_meta = meta_candidates[0]
            for old in meta_candidates[1:]:
                os.remove(os.path.join(dirpath, old))
            target_meta = os.path.join(dirpath, f'{prefix}_meta.json')
            if latest_meta != f'{prefix}_meta.json':
                os.replace(os.path.join(dirpath, latest_meta), target_meta)

print('cleanup done')
