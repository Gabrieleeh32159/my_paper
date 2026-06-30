# Graph Report - my_paper  (2026-06-18)

## Corpus Check
- 57 files · ~1,233,700 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 441 nodes · 677 edges · 26 communities (25 shown, 1 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 9 edges (avg confidence: 0.8)
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
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]

## God Nodes (most connected - your core abstractions)
1. `evaluate_fusion_cv()` - 17 edges
2. `compute_fold_gains()` - 17 edges
3. `BaseDataset` - 16 edges
4. `descriptor_slices()` - 16 edges
5. `ndarray` - 15 edges
6. `Blueprint detallado — bloques LaTeX exactos para implementar` - 14 edges
7. `tsi_from_fold_gains()` - 13 edges
8. `load_progress()` - 12 edges
9. `save_progress()` - 12 edges
10. `permutation_null_kstar_gains()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `gate_threshold()`  [EXTRACTED]
  revision/compute_tsi_from_checkpoints.py → experiments/src/tsi.py
- `test_svm_subsample_only_early_fusion()` --calls--> `SVMClassifier`  [INFERRED]
  experiments/tests/test_method.py → experiments/src/classifiers.py
- `test_inverse_frequency_pos_weight_is_nontrivial_and_clipped()` --calls--> `inverse_frequency_pos_weight()`  [INFERRED]
  experiments/tests/test_method.py → experiments/src/classifiers.py
- `test_mlp_self_provisions_validation_M4()` --calls--> `MLPClassifier`  [INFERRED]
  experiments/tests/test_method.py → experiments/src/classifiers.py
- `test_mlp_mtat_factory_enables_weighted_bce()` --calls--> `make_clf_factory()`  [INFERRED]
  experiments/tests/test_method.py → experiments/src/classifiers.py

## Import Cycles
- None detected.

## Communities (26 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (27): ndarray, _MLPBase, _calibrated_base_estimators(), get_classifier(), inverse_frequency_pos_weight(), _make_rf_estimator(), MLPClassifier, MLPModel (+19 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (41): ndarray, atomic_write_json(), _json_default(), load_progress(), _normalize(), Lightweight, crash-safe checkpointing for the heavy CV drivers.  The TSI / fusio, JSON encoder for numpy scalars/arrays (mirrors the notebook's ``_default``)., Round-trip ``meta`` through JSON so comparison ignores tuple/list and     numpy/ (+33 more)

### Community 2 - "Community 2"
Cohesion: 0.08
Nodes (43): ndarray, make_clf_factory(), Build a classifier factory closure for the TSI / fusion drivers.      The driver, descriptor_slices(), extract_descriptor(), Column indices of each descriptor within the 192-d track vector.      The 192-d, Slice the sub-vector of a single descriptor out of a 192-d track matrix.      Pa, combine_late() (+35 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (15): BaseDataset, FMASmallDataset, get_dataset(), GTZANDataset, IRMASDataset, MagnaTagATuneDataset, Dataset loading utilities for TSI experiments.  Supports: GTZAN, FMA-small, Magn, Generate repeated stratified k-fold indices.          Returns list of (train_ind (+7 more)

### Community 4 - "Community 4"
Cohesion: 0.11
Nodes (31): ndarray, base_rate(), _binary_ece(), calibration_report(), chance_log_loss(), expected_calibration_error(), gain_from_predictions(), information_gain() (+23 more)

### Community 5 - "Community 5"
Cohesion: 0.10
Nodes (20): ndarray, build_tsi(), LRClf, main(), Generate small EXAMPLE result JSONs from synthetic data (NOT real experiments)., Calibrated logistic-regression stand-in mirroring the real pipeline.      Uses `, synth_dataset(), cliffs_delta() (+12 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (23): Blueprint detallado — bloques LaTeX exactos para implementar, C.10 Importancia / RQ huérfanas — [REEMPLAZA el encabezado de la subsección de importancia y su frase introductoria con RQ1/RQ3], C.11 Bonferroni 6→10 — [REEMPLAZA el párrafo "Entre estrategias de fusión:"], C.12 Spearman ρ>0.7 — [REEMPLAZA el párrafo final de consistencia Spearman], C.13 Limitaciones efecto piso — [REEMPLAZA la primera oración(es) del bloque de Limitaciones sobre dimensionalidad/efecto piso, conservando calibración e IRMAS], C.1 Abstract — [REEMPLAZA el `\begin{abstract}...\end{abstract}` completo], C.2 Intro — [REEMPLAZA el párrafo que empieza "Este trabajo aborda esta brecha introduciendo..." y la pregunta no; mantener la pregunta central], C.3 Intro contribuciones — [REEMPLAZA ítem 1 y ítem 3 de la enumerate] (+15 more)

### Community 7 - "Community 7"
Cohesion: 0.15
Nodes (19): ndarray, Standalone worker script for parallel feature extraction. Each worker processes, Process a chunk of tracks and save results to a worker-specific file., run_worker(), aggregate_window(), extract_frame_features(), extract_multiscale_features(), extract_multiscale_per_descriptor() (+11 more)

### Community 8 - "Community 8"
Cohesion: 0.13
Nodes (14): Calibración, Clasificadores (4), Contexto, Criterios de aceptación, Datasets y evaluación — `src/data_loader.py`, notebooks, Entregables, Especificación del método (resumida; el detalle pleno está en `paper/proposal.tex`), Estrategias de fusión (5) — `src/fusion.py` (+6 more)

### Community 9 - "Community 9"
Cohesion: 0.18
Nodes (13): ndarray, aggregate_importance_by_descriptor_and_scale(), bootstrap_importance_ci(), early_fusion_groups(), _f1_macro(), mdi_by_group(), permutation_importance_grouped(), Feature-importance analysis -- CORROBORATIVE only.  Per ``paper/proposal.tex`` ( (+5 more)

### Community 10 - "Community 10"
Cohesion: 0.19
Nodes (13): load_gains(), load_null_flat(), main(), Compute G, k*, tau, TSI from the downloaded XGBoost checkpoints.  Uses the exact, apply_gate(), Compute the (un-gated) TSI statistics for one descriptor.      Parameters     --, Apply the informativeness gate and exploitability rule to one descriptor.      `, tsi_from_fold_gains() (+5 more)

### Community 11 - "Community 11"
Cohesion: 0.20
Nodes (9): Fase 4 — Revisión adversarial independiente (paper final), M1 — Sesgo de selección definido de forma inconsistente en el TSI, M2 — `TSI_rel` inestable justo donde se la vende como solución, M3 — Los pesos de la TSI-weighted late fusion son heurísticos y no corresponden a lo que el clasificador usa, Novedad, Objeciones MAYORES (bloquean publicación), Objeciones MENORES, Para pasar a MINOR REVISION (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.22
Nodes (8): Experiments — Temporal Sensitivity Index (TSI), Extracted feature format (already computed — reuse, do NOT re-extract), How to run, Layout, Local development / tests, Method summary (see the paper for full detail), Paths (Google Colab — preserve literally), Protocol & reproducibility notes

### Community 13 - "Community 13"
Cohesion: 0.22
Nodes (8): Arreglo 1 — (PROTOCOLO, obligatorio) Cablear el submuestreo 50% de SVM en MTAT, Arreglo 2 — (cosmético) Default engañoso de profundidad en el wrapper RF de cuML, Arreglo 3 — (cosmético) Ejemplo sintético de IRMAS debe ilustrar K=2, Arreglo 4 — (reproducibilidad, doc) Registrar hiperparámetros de XGBoost, Contexto, Opcionales (no bloqueantes), Prompt para Claude Code — arreglos pendientes (post-auditoría), Reglas y criterios de aceptación

### Community 14 - "Community 14"
Cohesion: 0.46
Nodes (7): main(), Pretty-print the redesigned TSI / fusion results as Markdown tables.  Reads ``ts, Honest-estimator diagnostic: in-sample vs nested TSI payoff per descriptor., show_experiments(), show_residual_optimism(), show_tsi(), _table()

### Community 15 - "Community 15"
Cohesion: 0.25
Nodes (7): 1. Diagnóstico del problema del TSI, 2. Revisión por sección (resumen), 3. Consistencia interna, 4. Claims vs evidencia (suavizar), 5. Referencias, 6. Cambios priorizados, Fase 1 — Análisis (revisión de pares objetiva)

### Community 16 - "Community 16"
Cohesion: 0.25
Nodes (7): Confirmado coincidente con el paper, Correcciones verificadas (eran los puntos con impacto numérico), Discrepancias menores restantes, Reauditoría independiente (post-fixes) — código ↔ propuesta, Recomendación, Tests, Veredicto

### Community 17 - "Community 17"
Cohesion: 0.25
Nodes (7): Auditoría CIEGA independiente — código ↔ paper, Confirmado correcto e imparcial (coincide con el paper), Hallazgo nuevo e independiente, Otras menores (reporte/documentación), Recomendación, Tests, Veredicto

### Community 18 - "Community 18"
Cohesion: 0.25
Nodes (7): Arreglo 1 — (PROTOCOLO, obligatorio) Cablear *weighted binary cross-entropy* del MLP en MTAT, Arreglo 2 — (doc en el paper) Declarar que el TSI de MTAT usa CV, no la partición oficial, Arreglo 3 — (reporte, opcional) Significancia de la referencia learned-LF, Contexto, Opcional, Prompt para Claude Code — fixes tras la auditoría ciega, Reglas y criterios de aceptación

### Community 19 - "Community 19"
Cohesion: 0.29
Nodes (6): Cambios de texto (sin re-ejecutar), Decisión central del TSI, Fase 2 — Proposición (blueprint de reformulación), Notas de re-ejecución (quedan como % REVISAR en el .tex; el autor corre el código aparte), Rol mecánico real del TSI (nombres unificados), Saneamiento de `references.bib`

### Community 20 - "Community 20"
Cohesion: 0.29
Nodes (6): Arreglo 1 — (PROTOCOLO, obligatorio) Restringir el submuestreo 50% de SVM a early-fusion, Arreglo 2 — (corrección menor) `TSI_rel` acotado a [0,1], Arreglo 3 — (reporte) Exponer el sesgo optimista residual en el resumen, Contexto, Prompt para Claude Code — fixes tras la 2ª auditoría ciega (solo código), Reglas y criterios de aceptación

### Community 21 - "Community 21"
Cohesion: 0.29
Nodes (6): Confirmado correcto e imparcial, Confirmó los 3 fixes de la ronda anterior, Hallazgos restantes — todos paper↔código (ya cerrados en el paper), Lectura, Tercera auditoría CIEGA — código ↔ paper, Tests

### Community 22 - "Community 22"
Cohesion: 0.33
Nodes (5): Auditoría independiente — código `experiments/` ↔ `paper/proposal.tex`, Discrepancias menores (con fix), Lo que coincide con el paper (verificado, con archivo/función), Recomendación, Veredicto

### Community 23 - "Community 23"
Cohesion: 0.33
Nodes (5): Confirmado correcto e imparcial, Discrepancias menores (nuevas, halladas sin sesgo), Lectura, Segunda auditoría CIEGA — código ↔ paper, Tests

### Community 24 - "Community 24"
Cohesion: 0.40
Nodes (4): Bucle de revisión iterativa — registro de convergencia, El cambio que lo desbloqueó, Pendientes menores para cámara lista (no bloqueantes), Veredicto final

## Knowledge Gaps
- **105 isolated node(s):** `ndarray`, `_FitCounter`, `Layout`, `Extracted feature format (already computed — reuse, do NOT re-extract)`, `Paths (Google Colab — preserve literally)` (+100 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `test_svm_subsample_only_early_fusion()` connect `Community 2` to `Community 0`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Why does `SVMClassifier` connect `Community 0` to `Community 2`?**
  _High betweenness centrality (0.033) - this node is a cross-community bridge._
- **What connects `Generate small EXAMPLE result JSONs from synthetic data (NOT real experiments).`, `Calibrated logistic-regression stand-in mirroring the real pipeline.      Uses ``, `Pretty-print the redesigned TSI / fusion results as Markdown tables.  Reads ``ts` to the rest of the system?**
  _202 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.06289308176100629 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.07729468599033816 - nodes in this community are weakly interconnected._
- **Should `Community 2` be split into smaller, more focused modules?**
  _Cohesion score 0.07878787878787878 - nodes in this community are weakly interconnected._
- **Should `Community 3` be split into smaller, more focused modules?**
  _Cohesion score 0.059800664451827246 - nodes in this community are weakly interconnected._