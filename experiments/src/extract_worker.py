"""
Standalone worker script for parallel feature extraction.
Each worker processes a chunk of tracks and saves results independently.
Called as: python _extract_worker.py <args_json_path>

This avoids fork/spawn issues entirely since each worker is a fresh process.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path


def run_worker(worker_id, filepaths, indices, output_dir, sr, dataset_name):
    """Process a chunk of tracks and save results to a worker-specific file."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.features import extract_multiscale_features, load_and_preprocess

    output_dir = Path(output_dir)
    worker_file = output_dir / f"{dataset_name}_worker_{worker_id}.npz"
    progress_file = output_dir / f"{dataset_name}_worker_{worker_id}_progress.json"

    # Resume from checkpoint
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
                print(f"[Worker {worker_id}] Resuming from {start_pos}/{len(filepaths)}", flush=True)
        except Exception as e:
            print(f"[Worker {worker_id}] Could not resume: {e}", flush=True)
            start_pos = 0
            all_short, all_medium, all_long = [], [], []
            completed_indices, errors = [], []

    total = len(filepaths)
    print(f"[Worker {worker_id}] Starting: {total - start_pos} tracks to process", flush=True)

    for pos in range(start_pos, total):
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

        # Checkpoint every 50 tracks
        if (pos + 1) % 50 == 0 or pos == total - 1:
            np.savez(
                worker_file,
                short=np.stack(all_short) if all_short else np.empty((0, 192)),
                medium=np.stack(all_medium) if all_medium else np.empty((0, 192)),
                long=np.stack(all_long) if all_long else np.empty((0, 192)),
                indices=np.array(completed_indices),
            )
            with open(progress_file, 'w') as f:
                json.dump({'completed': pos + 1, 'total': total, 'errors': errors,
                           'n_ok': len(completed_indices)}, f)

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
        json.dump({'completed': total, 'total': total, 'done': True,
                   'errors': errors, 'n_ok': len(completed_indices)}, f)

    print(f"[Worker {worker_id}] Done: {len(completed_indices)} OK, {len(errors)} errors", flush=True)


if __name__ == '__main__':
    args_path = sys.argv[1]
    with open(args_path) as f:
        args = json.load(f)
    run_worker(**args)
