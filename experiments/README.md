# Experiments — Temporal Sensitivity Index (TSI)

Implementation of the **redesigned (payoff)** TSI method. The source of truth for
the method is `paper/proposal.tex`; this folder is the executable companion. Any
older code (dispersion `std_k G`, Wilcoxon best-vs-worst, multiplicative TSI
weights) has been removed — the backup lives in `../experiments_legacy.zip`.

## Layout

```
experiments/
├── src/
│   ├── features.py       # multi-scale extraction + descriptor slicing helpers
│   ├── data_loader.py    # GTZAN / FMA-small / MTAT / IRMAS loaders
│   ├── classifiers.py    # XGBoost, RF, SVM, MLP (calibrated) + multilabel wrapper
│   ├── evaluation.py     # G = 1 - L/L_chance, L_chance, ECE, reliability curves
│   ├── tsi.py            # nested k*, payoff TSI, gate τ, bootstrap CI, TSI_rel
│   ├── fusion.py         # the 5 fusion strategies (learned LF weights)
│   ├── stats.py          # Friedman/Nemenyi, Wilcoxon+Bonferroni, Cliff, Spearman
│   ├── importance.py     # PI / MDI (corroborative only)
│   └── extract_worker.py # optional parallel extraction worker
├── tests/test_method.py  # synthetic-data unit tests (no Drive needed)
├── results/
│   ├── show.py                 # prints results as Markdown tables
│   ├── _gen_example_results.py # regenerates the example JSONs below
│   ├── tsi_results.json        # EXAMPLE output (synthetic; schema reference)
│   ├── experiment_results.json # EXAMPLE output (synthetic; schema reference)
│   └── run_config.json         # reproducibility metadata (written by notebook 02)
├── 01_feature_extraction.ipynb # Colab: extract features (idempotent)
└── 02_tsi_and_fusion.ipynb     # Colab: TSI + fusion + stats → results JSON
```

## Extracted feature format (already computed — reuse, do NOT re-extract)

Per dataset `ds ∈ {gtzan, fma_small, mtat, irmas}` in `FEATURES_ROOT`:

- `{ds}_short.npy`, `{ds}_medium.npy`, `{ds}_long.npy`: `(n_tracks, 192)` float32,
  **raw / not standardized**.
- `{ds}_indices.npy`, `{ds}_labels.npy` (object; MTAT = 50-tag multilabel),
  `{ds}_splits.npy`, `{ds}_errors.json`. All rows aligned by position.

**192-d layout:**
`[ mean(window_means)·48 | mean(window_stds)·48 | std(window_means)·48 | std(window_stds)·48 ]`.
Within each 48-block the descriptor order is `FEATURE_DIMS`
(`mfcc 20, chroma 12, spectral_centroid 1, spectral_contrast 7, spectral_rolloff 1, zcr 1, tonnetz 6`).
A descriptor's sub-vector at one scale is its slice in each of the 4 blocks →
`4 × dim` (MFCC=80, ZCR=4). Use `features.extract_descriptor(matrix_192, name)`
and `features.descriptor_slices()`.

**Standardize per fold (fit on train only); never bake z-score into saved
features.** The classifier wrappers in `classifiers.py` standardize internally.

## Paths (Google Colab — preserve literally)

```
DRIVE_ROOT    = /content/drive/MyDrive/tsi_experiments
DATA_ROOT     = DRIVE_ROOT/data        # audio
FEATURES_ROOT = DRIVE_ROOT/features    # extracted features (format above)
RESULTS_ROOT  = DRIVE_ROOT/results     # JSON outputs (mirrored to results/)
```

## How to run

Both notebooks (1) mount Drive, (2) clone this repo to `/content/my_paper`,
(3) `sys.path.insert(0, '/content/my_paper/experiments')` and `from src.X import …`.

1. **`01_feature_extraction.ipynb`** — extracts the 7 descriptors at 3 scales and
   writes the `.npy` files above. **Idempotent**: it skips any dataset whose
   feature files already exist, so the features already on Drive remain valid and
   this notebook only re-runs if something is missing.
2. **`02_tsi_and_fusion.ipynb`** — loads the cached features and runs the full
   redesigned analysis: the per-descriptor TSI (nested `k*`, gate `τ`, payoff
   bootstrap CI, `TSI_rel`, residual optimism), the 7×3 gain matrix, the 5 fusion
   strategies, and the statistical validation. It writes `tsi_results.json` and
   `experiment_results.json` to `RESULTS_ROOT` and mirrors them to `results/`.

### Method summary (see the paper for full detail)

```
G(f,k)    = 1 − L(f,k) / L_chance              # normalized information gain
k*(f)     = argmax_k G(f,k)   (NESTED: chosen on inner folds, scored on outer)
TSI(f)    = [ G(f,k*) − G(f,k̄) ] · 1[ G(f,k*) > τ ]      with k̄ = medium (2 s)
TSI_rel(f)= ( G(f,k*) − G(f,k̄) ) / G(f,k*)     # secondary, gated only
```

`τ` is calibrated **per task** from a label-permutation null that re-selects the
argmax in each replicate. A descriptor is *temporally exploitable* iff the lower
bound of the payoff bootstrap CI `> 0` **and** `G(f,k*) > τ`. `k*` is defined
exactly **once** (`tsi.select_k_star`) and reused identically in the point
estimate, CI, gate and fusion. IRMAS uses **K=2** (long scale collapses);
Friedman is replaced by Wilcoxon and the TSI is directional toward the short scale.

## Protocol & reproducibility notes

- **Evaluation protocol.** The **TSI uses cross-validation for every dataset** by
  necessity — the gate `τ` (permutation null), the Friedman test and the payoff
  bootstrap CI all require multiple folds. GTZAN uses 5×10 repeated stratified CV;
  the official-split datasets use a stratified K-fold for that purpose. For the
  **final fusion point estimate** on FMA/MTAT/IRMAS the notebook *also* honours the
  **official partition** (`USE_OFFICIAL_SPLIT_FOR_FUSION`); the powered
  between-strategy Wilcoxon comes from GTZAN's repeated CV, since a single official
  fold cannot support a per-fold paired test.
- **Calibration reporting.** ECE is **averaged over all outer folds** and the pooled
  **reliability curve** is persisted (`experiment_results[ds][clf]['calibration']`).
- **Classifier reproducibility.** `run_config.json` records the run knobs, the fold
  scheme per dataset, the **RF backend actually used** (sklearn vs cuML — same
  500 trees / depth 30 either way), and the **XGBoost hyperparameters**
  (`n_estimators=500, max_depth=6, learning_rate=0.1`), which the paper leaves
  unfixed. RF/XGB/MLP expose a `.hyperparams` attribute.
- **MLP calibration.** When trained through the generic driver path
  (`fit(X, y)` without a validation set), the MLP **self-provisions a 15 % internal
  validation split** so early stopping and temperature scaling still apply
  (otherwise the temperature would remain 1.0, leaving probabilities uncalibrated).

## Local development / tests

The analysis modules do not require `librosa` (only notebook 01 does). Run the
unit tests on synthetic data:

```bash
cd experiments
python tests/test_method.py        # or: python -m pytest tests/ -q
python results/_gen_example_results.py   # regenerate example JSONs
python results/show.py                   # render the example tables
```
