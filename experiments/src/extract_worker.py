"""
Standalone worker script for parallel feature extraction.
Each worker processes a chunk of tracks and saves results independently.
Can be called as: python -m src.extract_worker <args_json_path>

This avoids fork/spawn issues entirely since each worker is a fresh process.
"""

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm


def run_worker(worker_id, filepaths, indices, output_dir, sr, dataset_name):
    """Process a chunk of tracks and save results to a worker-specific file."""
    from src.features import extract_multiscale_features, load_and_preprocess

    output_dir = Path(output_dir)
    worker_file = output_dir / f"{dataset_name}_worker_{worker_id}.npz"
    progress_file = output_dir / f"{dataset_name}_worker_{worker_id}_progress.json"

    # Check for existing progress
    start_pos = 0
    all_short, all_medium, all_long = [], [], []
    completed_indices = []
    errors = []

    if worker_file.exists() and progress_file.exists():
        try:
            with open(progress_file) as f:
                progress = json.load(f)
            start_pos = progress.get('completed', 0)
            if start_pos > 0:
                data = np.load(worker_file, allow_pickle=True)
                all_short = list(data['short'])
                all_medium = list(data['medium'])
                all_long = list(data['long'])
                completed_indices = list(data['indices'])
                errors = progress.get('errors', [])
                print(f"  [Worker {worker_id}] Resuming from {start_pos}/{len(filepaths)}")
        except Exception:
            start_pos = 0
            all_short, all_medium, all_long = [], [], []
            completed_indices = []
            errors = []

    total = len(filepaths)
    desc = f"Worker {worker_id}"

    for pos in tqdm(range(start_pos, total), desc=desc, initial=start_pos, total=total,
                    position=worker_id, leave=True):
        filepath = filepaths[pos]
        idx = indices[pos]

        try:
            y = load_and_preprocess(filepath, sr=sr)
            if len(y) < sr:
                errors.append((int(idx), str(filepath), "too short"))
                continue

            feats = extract_multiscale_features(y, sr=sr)
            all_short.append(feats['short'])
            all_medium.append(feats['medium'])
            all_long.append(feats['long'])
            completed_indices.append(int(idx))
            del y, feats

        except Exception as e:
            errors.append((int(idx), str(filepath), str(e)))
            continue

        # Save checkpoint every 100 tracks
        if (pos + 1) % 100 == 0 or pos == total - 1:
            np.savez(
                worker_file,
                short=np.stack(all_short) if all_short else np.empty((0, 192)),
                medium=np.stack(all_medium) if all_medium else np.empty((0, 192)),
                long=np.stack(all_long) if all_long else np.empty((0, 192)),
                indices=np.array(completed_indices),
            )
            with open(progress_file, 'w') as f:
                json.dump({'completed': pos + 1, 'total': total, 'errors': errors}, f)

    # Final save
    if all_short:
        np.savez(
            worker_file,
            short=np.stack(all_short),
            medium=np.stack(all_medium),
            long=np.stack(all_long),
            indices=np.array(completed_indices),
        )
    with open(progress_file, 'w') as f:
        json.dump({'completed': total, 'total': total, 'done': True, 'errors': errors}, f)

    print(f"  [Worker {worker_id}] Done: {len(completed_indices)} OK, {len(errors)} errors")


if __name__ == '__main__':
    args_path = sys.argv[1]
    with open(args_path) as f:
        args = json.load(f)
    run_worker(**args)
