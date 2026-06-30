# Graph Report - experiments  (2026-06-22)

## Corpus Check
- 21 files · ~21,255 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 319 nodes · 619 edges · 12 communities (11 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a1134c48`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]

## God Nodes (most connected - your core abstractions)
1. `compute_fold_gains()` - 20 edges
2. `descriptor_slices()` - 19 edges
3. `evaluate_fusion_cv()` - 19 edges
4. `ndarray` - 17 edges
5. `BaseDataset` - 16 edges
6. `tsi_from_fold_gains()` - 14 edges
7. `permutation_null_kstar_gains()` - 14 edges
8. `load_progress()` - 13 edges
9. `save_progress()` - 13 edges
10. `calibration_report()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `test_chance_log_loss_uniform_is_logC()` --calls--> `chance_log_loss()`  [EXTRACTED]
  tests/test_method.py → src/evaluation.py
- `test_progress_iter_noop_and_order_preserving()` --calls--> `progress_iter()`  [EXTRACTED]
  tests/test_method.py → src/progress.py
- `test_select_k_star_tiebreak_is_deterministic()` --calls--> `select_k_star()`  [EXTRACTED]
  tests/test_method.py → src/tsi.py
- `synth_dataset()` --calls--> `descriptor_slices()`  [EXTRACTED]
  results/_gen_example_results.py → src/features.py
- `build_tsi()` --calls--> `apply_gate()`  [EXTRACTED]
  results/_gen_example_results.py → src/tsi.py

## Import Cycles
- None detected.

## Communities (12 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (29): _MLPBase, _calibrated_base_estimators(), _dense_relabel(), _expand_proba(), inverse_frequency_pos_weight(), _make_rf_estimator(), MLPClassifier, MLPModel (+21 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (48): atomic_write_json(), _json_default(), load_progress(), _normalize(), Lightweight, crash-safe checkpointing for the heavy CV drivers.  The TSI / fusio, JSON encoder for numpy scalars/arrays (mirrors the notebook's ``_default``)., Round-trip ``meta`` through JSON so comparison ignores tuple/list and     numpy/, Write ``obj`` to ``path`` atomically (``*.tmp`` then :func:`os.replace`). (+40 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (15): BaseDataset, FMASmallDataset, get_dataset(), GTZANDataset, IRMASDataset, MagnaTagATuneDataset, Dataset loading utilities for TSI experiments.  Supports: GTZAN, FMA-small, Magn, Generate repeated stratified k-fold indices.          Returns list of (train_ind (+7 more)

### Community 3 - "Community 3"
Cohesion: 0.09
Nodes (34): make_clf_factory(), Build a classifier factory closure for the TSI / fusion drivers.      The driver, combine_late(), early_fusion(), learn_late_fusion_weights(), ndarray, Weighted average of per-scale probability matrices., Concatenate per-scale 192-d vectors -> ``(n, 192*len(scales))``. (+26 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (30): Standalone worker script for parallel feature extraction. Each worker processes, Process a chunk of tracks and save results to a worker-specific file., run_worker(), aggregate_window(), descriptor_slices(), extract_descriptor(), extract_frame_features(), extract_multiscale_features() (+22 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (28): base_rate(), _binary_ece(), chance_log_loss(), expected_calibration_error(), gain_from_predictions(), information_gain(), log_loss_multiclass(), log_loss_multilabel() (+20 more)

### Community 6 - "Community 6"
Cohesion: 0.10
Nodes (23): build_tsi(), LRClf, main(), Generate small EXAMPLE result JSONs from synthetic data (NOT real experiments)., Calibrated logistic-regression stand-in mirroring the real pipeline.      Uses `, synth_dataset(), cliffs_delta(), cliffs_magnitude() (+15 more)

### Community 7 - "Community 7"
Cohesion: 0.18
Nodes (13): aggregate_importance_by_descriptor_and_scale(), bootstrap_importance_ci(), early_fusion_groups(), _f1_macro(), mdi_by_group(), permutation_importance_grouped(), ndarray, Feature-importance analysis -- CORROBORATIVE only.  Per ``paper/proposal.tex`` ( (+5 more)

### Community 8 - "Community 8"
Cohesion: 0.22
Nodes (8): Experiments — Temporal Sensitivity Index (TSI), Extracted feature format (already computed — reuse, do NOT re-extract), How to run, Layout, Local development / tests, Method summary (see the paper for full detail), Paths (Google Colab — preserve literally), Protocol & reproducibility notes

### Community 9 - "Community 9"
Cohesion: 0.39
Nodes (8): get_classifier(), Factory function to create a classifier by name.      Parameters     ----------, Classifiers must tolerate a training set whose labels are a non-contiguous subse, Synthetic data whose labels are exactly ``present_labels`` (a subset of     the, _subset_data(), test_full_class_set_unchanged(), test_rf_predict_proba_is_global_width_on_subset(), test_xgb_fits_on_noncontiguous_label_subset()

### Community 10 - "Community 10"
Cohesion: 0.46
Nodes (7): main(), Pretty-print the redesigned TSI / fusion results as Markdown tables.  Reads ``ts, Honest-estimator diagnostic: in-sample vs nested TSI payoff per descriptor., show_experiments(), show_residual_optimism(), show_tsi(), _table()

## Knowledge Gaps
- **8 isolated node(s):** `ndarray`, `_FitCounter`, `Layout`, `Extracted feature format (already computed — reuse, do NOT re-extract)`, `Paths (Google Colab — preserve literally)` (+3 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MultiLabelWrapper` connect `Community 0` to `Community 9`, `Community 3`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `MLPClassifier` connect `Community 0` to `Community 9`, `Community 3`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `descriptor_slices()` connect `Community 4` to `Community 1`, `Community 3`, `Community 6`, `Community 7`?**
  _High betweenness centrality (0.056) - this node is a cross-community bridge._
- **What connects `Generate small EXAMPLE result JSONs from synthetic data (NOT real experiments).`, `Calibrated logistic-regression stand-in mirroring the real pipeline.      Uses ``, `Pretty-print the redesigned TSI / fusion results as Markdown tables.  Reads ``ts` to the rest of the system?**
  _108 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06038961038961039 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.07896575821104122 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.059800664451827246 - nodes in this community are weakly interconnected._