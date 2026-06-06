"""
Multi-scale audio feature extraction module.

Extracts 7 canonical audio descriptors at 3 temporal scales (200ms, 2s, 5s)
using librosa. Aggregates frame-level features into fixed-length vectors
via mean + std statistics.
"""

import numpy as np
from typing import Dict, Tuple, Optional

# ``librosa`` is only needed for audio extraction (notebook 01). It is imported
# lazily inside the extraction functions so that the analysis modules (TSI,
# fusion, stats) and the unit tests can import the slicing helpers without it.


# Constants
SR = 16000
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 20
N_CHROMA = 12
N_CONTRAST_BANDS = 7
N_TONNETZ = 6

# Temporal scales (in seconds)
SCALES = {'short': 0.2, 'medium': 2.0, 'long': 5.0}

# Feature dimensions
FEATURE_DIMS = {
    'mfcc': N_MFCC,          # 20
    'chroma': N_CHROMA,       # 12
    'spectral_centroid': 1,   # 1
    'spectral_contrast': N_CONTRAST_BANDS,  # 7
    'spectral_rolloff': 1,    # 1
    'zcr': 1,                 # 1
    'tonnetz': N_TONNETZ,     # 6
}

TOTAL_FRAME_FEATURES = sum(FEATURE_DIMS.values())  # 48
AGGREGATED_DIM = TOTAL_FRAME_FEATURES * 2  # 96 (mean + std)
TRACK_DIM = AGGREGATED_DIM * 2  # 192 (mean + std across windows)


def extract_frame_features(y: np.ndarray, sr: int = SR) -> Dict[str, np.ndarray]:
    """
    Extract all 7 frame-level descriptors from an audio signal.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (mono, resampled to sr).
    sr : int
        Sample rate.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping descriptor name to frame-level feature matrix.
        Each value has shape (n_features, n_frames).
    """
    import librosa  # lazy: only needed for audio extraction
    features = {}

    # MFCCs (20, n_frames)
    features['mfcc'] = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH, n_fft=N_FFT
    )

    # Chroma (12, n_frames)
    features['chroma'] = librosa.feature.chroma_stft(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )

    # Spectral Centroid (1, n_frames)
    features['spectral_centroid'] = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )

    # Spectral Contrast (7, n_frames)
    features['spectral_contrast'] = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
    )

    # Spectral Rolloff (1, n_frames)
    features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
        y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT, roll_percent=0.85
    )

    # Zero-Crossing Rate (1, n_frames)
    features['zcr'] = librosa.feature.zero_crossing_rate(
        y=y, hop_length=HOP_LENGTH
    )

    # Tonnetz (6, n_frames)
    features['tonnetz'] = librosa.feature.tonnetz(
        y=y, sr=sr
    )

    return features


def aggregate_window(frame_features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Aggregate frame-level features within a single window using mean + std.

    Parameters
    ----------
    frame_features : Dict[str, np.ndarray]
        Frame-level features, each with shape (n_features, n_frames).

    Returns
    -------
    np.ndarray
        Aggregated vector of shape (96,) = 48 means + 48 stds.
    """
    means = []
    stds = []

    for name in FEATURE_DIMS.keys():
        feat = frame_features[name]  # (n_features, n_frames)
        means.append(np.mean(feat, axis=1))
        stds.append(np.std(feat, axis=1))

    return np.concatenate([np.concatenate(means), np.concatenate(stds)])


def split_into_windows(y: np.ndarray, window_sec: float, sr: int = SR) -> list:
    """
    Split audio signal into non-overlapping windows.

    Parameters
    ----------
    y : np.ndarray
        Audio signal.
    window_sec : float
        Window duration in seconds.
    sr : int
        Sample rate.

    Returns
    -------
    list of np.ndarray
        List of audio segments.
    """
    window_samples = int(window_sec * sr)
    n_windows = len(y) // window_samples

    if n_windows == 0:
        # If audio is shorter than window, use the full audio as one window
        return [y]

    windows = []
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        windows.append(y[start:end])

    return windows


def extract_multiscale_features(
    y: np.ndarray,
    sr: int = SR,
    scales: Optional[Dict[str, float]] = None
) -> Dict[str, np.ndarray]:
    """
    Extract features at multiple temporal scales for a single track.

    For each scale:
    1. Split audio into non-overlapping windows of that duration.
    2. Extract frame-level features within each window.
    3. Aggregate (mean+std) within each window → 96-d vector per window.
    4. Aggregate (mean+std) across windows → 192-d vector per scale.

    Parameters
    ----------
    y : np.ndarray
        Audio signal (mono, resampled to sr).
    sr : int
        Sample rate.
    scales : dict, optional
        Mapping of scale names to durations in seconds.
        Defaults to {'short': 0.2, 'medium': 2.0, 'long': 5.0}.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping of scale name to track-level feature vector (192-d each).
    """
    if scales is None:
        scales = SCALES

    result = {}

    for scale_name, window_sec in scales.items():
        windows = split_into_windows(y, window_sec, sr)

        window_vectors = []
        for w in windows:
            if len(w) < N_FFT:
                continue  # Skip windows shorter than FFT size
            frame_feats = extract_frame_features(w, sr)
            agg = aggregate_window(frame_feats)
            window_vectors.append(agg)

        if len(window_vectors) == 0:
            # Fallback: extract from full audio
            frame_feats = extract_frame_features(y, sr)
            agg = aggregate_window(frame_feats)
            window_vectors = [agg]

        window_matrix = np.stack(window_vectors)  # (n_windows, 96)

        # Aggregate across windows: mean + std → 192-d
        track_mean = np.mean(window_matrix, axis=0)
        track_std = np.std(window_matrix, axis=0)
        result[scale_name] = np.concatenate([track_mean, track_std])

    return result


def extract_multiscale_per_descriptor(
    y: np.ndarray,
    sr: int = SR,
    scales: Optional[Dict[str, float]] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Extract features per descriptor per scale (for TSI computation).

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Nested dict: descriptor_name -> scale_name -> feature vector.
        Each feature vector has shape (2 * descriptor_dim,) for window-level,
        then (4 * descriptor_dim,) for track-level (mean+std of mean+std).
    """
    if scales is None:
        scales = SCALES

    result = {desc: {} for desc in FEATURE_DIMS.keys()}

    for scale_name, window_sec in scales.items():
        windows = split_into_windows(y, window_sec, sr)

        # Collect per-descriptor aggregations across windows
        desc_windows = {desc: [] for desc in FEATURE_DIMS.keys()}

        for w in windows:
            if len(w) < N_FFT:
                continue
            frame_feats = extract_frame_features(w, sr)
            for desc in FEATURE_DIMS.keys():
                feat = frame_feats[desc]  # (dim, n_frames)
                desc_mean = np.mean(feat, axis=1)
                desc_std = np.std(feat, axis=1)
                desc_windows[desc].append(np.concatenate([desc_mean, desc_std]))

        # Fallback
        if all(len(v) == 0 for v in desc_windows.values()):
            frame_feats = extract_frame_features(y, sr)
            for desc in FEATURE_DIMS.keys():
                feat = frame_feats[desc]
                desc_mean = np.mean(feat, axis=1)
                desc_std = np.std(feat, axis=1)
                desc_windows[desc].append(np.concatenate([desc_mean, desc_std]))

        # Aggregate across windows per descriptor
        for desc in FEATURE_DIMS.keys():
            mat = np.stack(desc_windows[desc])  # (n_windows, 2*dim)
            track_mean = np.mean(mat, axis=0)
            track_std = np.std(mat, axis=0)
            result[desc][scale_name] = np.concatenate([track_mean, track_std])

    return result


def get_descriptor_indices() -> Dict[str, Tuple[int, int]]:
    """
    Get the start and end indices of each descriptor in the 192-d vector.

    The 192-d vector is structured as:
    [mean_of_window_means (48) | mean_of_window_stds (48) |
     std_of_window_means (48) | std_of_window_stds (48)]

    Within each 48-d block, descriptors are ordered as in FEATURE_DIMS.

    Returns
    -------
    Dict[str, Tuple[int, int]]
        Mapping of descriptor name to (start_idx, end_idx) in the 96-d
        window aggregation. For the full 192-d, these indices apply to
        both the first and second halves.
    """
    indices = {}
    offset = 0
    for name, dim in FEATURE_DIMS.items():
        indices[name] = (offset, offset + dim)
        offset += dim
    return indices


# Number of 48-d statistic blocks composing the 192-d track vector:
# [ mean(window_means) | mean(window_stds) | std(window_means) | std(window_stds) ]
N_STAT_BLOCKS = TRACK_DIM // TOTAL_FRAME_FEATURES  # 192 / 48 = 4


def descriptor_slices() -> Dict[str, np.ndarray]:
    """
    Column indices of each descriptor within the 192-d track vector.

    The 192-d vector is four stacked 48-d blocks (see ``get_descriptor_indices``).
    A descriptor occupies the SAME relative slice inside every block, so its full
    set of columns is its 48-block slice replicated at offsets 0, 48, 96, 144.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping descriptor name -> int array of length ``4 * dim`` with the
        absolute column indices of that descriptor in the 192-d vector.
        For MFCC this has length 80, for ZCR length 4, etc.
    """
    block_indices = get_descriptor_indices()
    slices = {}
    for name, (start, end) in block_indices.items():
        cols = []
        for b in range(N_STAT_BLOCKS):
            base = b * TOTAL_FRAME_FEATURES
            cols.extend(range(base + start, base + end))
        slices[name] = np.asarray(cols, dtype=int)
    return slices


def extract_descriptor(matrix_192: np.ndarray, descriptor_name: str) -> np.ndarray:
    """
    Slice the sub-vector of a single descriptor out of a 192-d track matrix.

    Parameters
    ----------
    matrix_192 : np.ndarray
        Array of shape ``(n_tracks, 192)`` (or ``(192,)`` for a single track),
        following the layout documented in ``descriptor_slices``.
    descriptor_name : str
        One of the keys of ``FEATURE_DIMS``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_tracks, 4 * dim)`` (or ``(4 * dim,)`` for 1-D input)
        with only the columns belonging to ``descriptor_name``.
    """
    if descriptor_name not in FEATURE_DIMS:
        raise ValueError(
            f"Unknown descriptor {descriptor_name!r}; "
            f"choose from {list(FEATURE_DIMS.keys())}"
        )
    cols = descriptor_slices()[descriptor_name]
    mat = np.asarray(matrix_192)
    if mat.ndim == 1:
        return mat[cols]
    return mat[:, cols]


def load_and_preprocess(filepath: str, sr: int = SR, top_db: float = 20.0) -> np.ndarray:
    """
    Load audio file, resample to mono at target sr, and trim silence.

    Parameters
    ----------
    filepath : str
        Path to audio file.
    sr : int
        Target sample rate.
    top_db : float
        Threshold for silence trimming.

    Returns
    -------
    np.ndarray
        Preprocessed audio signal.
    """
    import librosa  # lazy: only needed for audio extraction
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    return y
