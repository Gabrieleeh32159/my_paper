"""
Lightweight, crash-safe checkpointing for the heavy CV drivers.

The TSI / fusion sweep trains tens of thousands of calibrated classifiers per
run (e.g. GTZAN alone: ~42k fits in :func:`tsi.compute_fold_gains` and ~52k in
:func:`tsi.permutation_null_kstar_gains`). A Colab session can be killed at any
moment, so the drivers persist their progress *as they go* and resume from disk
on the next run.

Design contract:

* A checkpoint file is JSON with shape ``{"meta": {...}, "data": {...}}``.
  ``meta`` is a small **config signature** (seed, scales, fold count, ...); the
  driver only resumes from ``data`` when the stored ``meta`` matches what it is
  about to compute, otherwise it starts fresh (so changing a knob never silently
  mixes incompatible folds).
* ``data`` is keyed by ``str(fold_index)`` (JSON requires string keys); the
  driver decides the inner structure (per descriptor, per strategy, ...).
* Writes are **atomic** (write to ``*.tmp`` then :func:`os.replace`), so a kill
  mid-write cannot corrupt an existing checkpoint.

All helpers are no-ops when ``path is None`` so the drivers stay byte-for-byte
identical to their pre-checkpoint behaviour (the test suite never passes a path).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np


def _json_default(o):
    """JSON encoder for numpy scalars/arrays (mirrors the notebook's ``_default``)."""
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _normalize(meta: dict) -> dict:
    """Round-trip ``meta`` through JSON so comparison ignores tuple/list and
    numpy/native distinctions (a stored signature is always JSON-native)."""
    return json.loads(json.dumps(meta, default=_json_default))


def atomic_write_json(path, obj) -> None:
    """Write ``obj`` to ``path`` atomically (``*.tmp`` then :func:`os.replace`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, default=_json_default)
    os.replace(tmp, path)  # atomic on POSIX and on the Drive FUSE mount


def load_progress(path, meta: dict) -> dict:
    """Return the saved ``data`` dict if its ``meta`` matches, else ``{}``.

    Returns ``{}`` when ``path`` is ``None``/missing/corrupt, or when the stored
    config signature differs from ``meta`` (stale checkpoint -> recompute fresh).
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            blob = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(blob, dict) or blob.get("meta") != _normalize(meta):
        return {}
    data = blob.get("data")
    return data if isinstance(data, dict) else {}


def save_progress(path, meta: dict, data: dict) -> None:
    """Persist ``{"meta": meta, "data": data}`` atomically. No-op when ``path`` is None."""
    if path is None:
        return
    atomic_write_json(path, {"meta": _normalize(meta), "data": data})
