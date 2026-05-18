"""
Dataset loading utilities for TSI experiments.

Supports: GTZAN, FMA-small, MagnaTagATune, IRMAS.
All datasets are expected to be pre-downloaded on Google Drive.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from pathlib import Path


class BaseDataset:
    """Base class for all datasets."""

    def __init__(self, root: str):
        self.root = Path(root)
        self._paths = []
        self._labels = []
        self._splits = []

    def __len__(self):
        return len(self._paths)

    def get_audio_path(self, idx: int) -> str:
        return str(self._paths[idx])

    def get_label(self, idx: int):
        return self._labels[idx]

    def get_split(self, idx: int) -> str:
        return self._splits[idx]

    def get_split_indices(self, split: str) -> List[int]:
        return [i for i, s in enumerate(self._splits) if s == split]

    @property
    def n_classes(self):
        raise NotImplementedError

    @property
    def task_type(self):
        """'multiclass' or 'multilabel'"""
        return 'multiclass'

    @property
    def class_names(self) -> List[str]:
        raise NotImplementedError


class GTZANDataset(BaseDataset):
    """
    GTZAN dataset (1000 tracks, 10 genres, 30s each).

    Expected structure:
        root/
        ├── genres/
        │   ├── blues/
        │   ├── classical/
        │   ├── country/
        │   ├── disco/
        │   ├── hiphop/
        │   ├── jazz/
        │   ├── metal/
        │   ├── pop/
        │   ├── reggae/
        │   └── rock/
        └── train_filtered.txt (optional, Kereliuk partition)
            test_filtered.txt
    """

    GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']

    def __init__(self, root: str, use_filtered: bool = True, n_folds: int = 10):
        super().__init__(root)
        self.n_folds = n_folds
        self._genre_to_idx = {g: i for i, g in enumerate(self.GENRES)}
        self._load(use_filtered)

    def _load(self, use_filtered: bool):
        genres_dir = self.root / 'genres'

        # Check for alternative structure (genres_original)
        if not genres_dir.exists():
            genres_dir = self.root / 'genres_original'

        for genre in self.GENRES:
            genre_dir = genres_dir / genre
            if not genre_dir.exists():
                continue
            for f in sorted(genre_dir.iterdir()):
                if f.suffix in ('.wav', '.au', '.mp3'):
                    self._paths.append(f)
                    self._labels.append(self._genre_to_idx[genre])

        # For GTZAN we use k-fold CV, assign all as 'train' initially
        self._splits = ['train'] * len(self._paths)

    def get_cv_folds(self, n_folds: int = 10, n_repeats: int = 5, seed: int = 42):
        """
        Generate repeated stratified k-fold indices.

        Returns list of (train_indices, test_indices) tuples.
        """
        from sklearn.model_selection import RepeatedStratifiedKFold
        rskf = RepeatedStratifiedKFold(
            n_splits=n_folds, n_repeats=n_repeats, random_state=seed
        )
        labels = np.array(self._labels)
        return list(rskf.split(np.zeros(len(labels)), labels))

    @property
    def n_classes(self):
        return 10

    @property
    def class_names(self):
        return self.GENRES


class FMASmallDataset(BaseDataset):
    """
    FMA-small dataset (8000 tracks, 8 genres, 30s each).

    Expected structure:
        root/
        ├── fma_small/
        │   ├── 000/
        │   ├── 001/
        │   └── ...
        └── fma_metadata/
            ├── tracks.csv
            └── genres.csv
    """

    def __init__(self, root: str):
        super().__init__(root)
        self._load()

    def _load(self):
        metadata_dir = self.root / 'fma_metadata'
        audio_dir = self.root / 'fma_small'

        # Load tracks metadata
        tracks = pd.read_csv(
            metadata_dir / 'tracks.csv',
            index_col=0,
            header=[0, 1]
        )

        # Get small subset
        small = tracks[tracks[('set', 'subset')] == 'small']

        for track_id, row in small.iterrows():
            # FMA file naming: {track_id:06d}.mp3
            tid_str = f"{int(track_id):06d}"
            folder = tid_str[:3]
            filepath = audio_dir / folder / f"{tid_str}.mp3"

            if not filepath.exists():
                continue

            genre = row[('track', 'genre_top')]
            split = row[('set', 'split')]

            self._paths.append(filepath)
            self._labels.append(genre)
            self._splits.append(split)  # 'training', 'validation', 'test'

        # Encode genre labels to integers
        self._genre_names = sorted(set(self._labels))
        self._genre_to_idx = {g: i for i, g in enumerate(self._genre_names)}
        self._labels = [self._genre_to_idx[g] for g in self._labels]

        # Normalize split names
        split_map = {'training': 'train', 'validation': 'val', 'test': 'test'}
        self._splits = [split_map.get(s, s) for s in self._splits]

    @property
    def n_classes(self):
        return len(self._genre_names)

    @property
    def class_names(self):
        return self._genre_names


class MagnaTagATuneDataset(BaseDataset):
    """
    MagnaTagATune dataset (25,863 clips, 50 binary tags, ~29s each).

    Expected structure:
        root/
        ├── mp3/
        │   ├── 0/ to f/
        │   └── ...
        ├── annotations_final.csv
        └── split/
            ├── train.txt
            ├── valid.txt
            └── test.txt
    """

    def __init__(self, root: str, top_k_tags: int = 50):
        super().__init__(root)
        self.top_k_tags = top_k_tags
        self._tag_names = []
        self._load()

    def _load(self):
        # Load annotations
        annotations_file = self.root / 'annotations_final.csv'
        if not annotations_file.exists():
            # Try alternative name
            annotations_file = self.root / 'annotations.csv'

        df = pd.read_csv(annotations_file, sep='\t')

        # Get top-k tags by frequency
        tag_cols = [c for c in df.columns if c not in ['clip_id', 'mp3_path']]
        tag_freqs = df[tag_cols].sum().sort_values(ascending=False)
        self._tag_names = tag_freqs.head(self.top_k_tags).index.tolist()

        # Load split information
        split_dir = self.root / 'split'
        split_map = {}

        for split_name, filename in [('train', 'train.txt'),
                                      ('val', 'valid.txt'),
                                      ('test', 'test.txt')]:
            split_file = split_dir / filename
            if split_file.exists():
                with open(split_file) as f:
                    for line in f:
                        clip_id = line.strip()
                        if clip_id:
                            split_map[clip_id] = split_name

        # Build dataset
        for _, row in df.iterrows():
            mp3_path = row.get('mp3_path', '')
            clip_id = row.get('clip_id', '')

            filepath = self.root / 'mp3' / mp3_path
            if not filepath.exists():
                filepath = self.root / mp3_path

            # Multi-label: vector of 0/1 for top-k tags
            label = np.array([row[t] for t in self._tag_names], dtype=np.float32)

            split = split_map.get(str(clip_id), 'train')

            self._paths.append(filepath)
            self._labels.append(label)
            self._splits.append(split)

    @property
    def n_classes(self):
        return self.top_k_tags

    @property
    def task_type(self):
        return 'multilabel'

    @property
    def class_names(self):
        return self._tag_names


class IRMASDataset(BaseDataset):
    """
    IRMAS dataset (6,705 fragments, 11 instruments, 3s each).

    Expected structure:
        root/
        ├── IRMAS-TrainingData/
        │   ├── cel/
        │   ├── cla/
        │   ├── flu/
        │   ├── gac/
        │   ├── gel/
        │   ├── org/
        │   ├── pia/
        │   ├── sax/
        │   ├── tru/
        │   ├── vio/
        │   └── voi/
        └── IRMAS-TestingData-Part1/
            └── Part1/
    """

    INSTRUMENTS = ['cel', 'cla', 'flu', 'gac', 'gel',
                   'org', 'pia', 'sax', 'tru', 'vio', 'voi']

    INSTRUMENT_NAMES = ['Cello', 'Clarinet', 'Flute', 'Acoustic Guitar',
                        'Electric Guitar', 'Organ', 'Piano', 'Saxophone',
                        'Trumpet', 'Violin', 'Voice']

    def __init__(self, root: str):
        super().__init__(root)
        self._instr_to_idx = {inst: i for i, inst in enumerate(self.INSTRUMENTS)}
        self._load()

    def _load(self):
        # Training data
        train_dir = self.root / 'IRMAS-TrainingData'
        if not train_dir.exists():
            train_dir = self.root / 'IRMAS-Training'

        for inst in self.INSTRUMENTS:
            inst_dir = train_dir / inst
            if not inst_dir.exists():
                continue
            for f in sorted(inst_dir.iterdir()):
                if f.suffix in ('.wav', '.mp3', '.ogg'):
                    self._paths.append(f)
                    self._labels.append(self._instr_to_idx[inst])
                    self._splits.append('train')

        # Testing data
        test_dir = self.root / 'IRMAS-TestingData-Part1'
        if not test_dir.exists():
            test_dir = self.root / 'IRMAS-Testing'

        if test_dir.exists():
            for f in sorted(test_dir.rglob('*.wav')):
                # Labels are in .txt files with same name
                label_file = f.with_suffix('.txt')
                if label_file.exists():
                    with open(label_file) as lf:
                        instruments = [line.strip() for line in lf if line.strip()]
                    # Use primary instrument (first listed)
                    if instruments and instruments[0] in self._instr_to_idx:
                        self._paths.append(f)
                        self._labels.append(self._instr_to_idx[instruments[0]])
                        self._splits.append('test')

    @property
    def n_classes(self):
        return 11

    @property
    def class_names(self):
        return self.INSTRUMENT_NAMES


def get_dataset(name: str, root: str, **kwargs) -> BaseDataset:
    """
    Factory function to instantiate a dataset by name.

    Parameters
    ----------
    name : str
        One of 'gtzan', 'fma_small', 'mtat', 'irmas'.
    root : str
        Root directory for the dataset.

    Returns
    -------
    BaseDataset
        Dataset instance.
    """
    datasets = {
        'gtzan': GTZANDataset,
        'fma_small': FMASmallDataset,
        'mtat': MagnaTagATuneDataset,
        'irmas': IRMASDataset,
    }
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list(datasets.keys())}")
    return datasets[name](root, **kwargs)
