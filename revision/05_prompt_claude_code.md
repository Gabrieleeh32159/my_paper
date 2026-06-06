# Prompt para Claude Code — reconstruir `experiments/` desde cero

> Copia todo lo que sigue (desde "Contexto") y pégalo en Claude Code, abierto en la raíz del repo `my_paper`.

---

## Contexto

Este repo acompaña un paper IEEE de Music Information Retrieval que introduce el **Temporal Sensitivity Index (TSI)**. La **fuente de verdad del método es `paper/proposal.tex`** (en español). Léelo antes de codear: las secciones de Metodología, definición del TSI, fusión y validación estadística describen el método EXACTO a implementar. Ignora el código actual en `experiments/` por completo: implementa una versión ANTERIOR y obsoleta del método (TSI como `std_k G`, Wilcoxon best-vs-worst, fusión con peso multiplicativo por TSI). NO reintroduzcas esas fórmulas legadas.

## Objetivo

Reconstruir desde cero la carpeta `experiments/` (módulos `src/` + **2 notebooks** para Google Colab) que implemente el método **rediseñado** del paper, reutilizando los features ya extraídos (no se reextrae audio). Primero **archiva** la carpeta actual a `experiments_legacy.zip` en la raíz y luego elimínala, para empezar limpio sin código legado.

## Workflow de ejecución (NO cambiar)

- Se ejecuta en **Google Colab**. Los notebooks: (1) montan Drive, (2) clonan este repo desde `https://github.com/Gabrieleeh32159/my_paper.git` a `/content/my_paper`, (3) `sys.path.insert(0, '/content/my_paper/experiments')` y hacen `from src.X import ...`.
- Rutas a **preservar** literalmente:
  - `DRIVE_ROOT = /content/drive/MyDrive/tsi_experiments`
  - `DATA_ROOT = DRIVE_ROOT/data`  (audio, ya descargado)
  - `FEATURES_ROOT = DRIVE_ROOT/features`  (features ya extraídos, ver formato abajo)
  - `RESULTS_ROOT = DRIVE_ROOT/results`  (escribir aquí los JSON; además copiar a `experiments/results/` del repo para versionar ejemplos)
- Mantener **exactamente 2 notebooks**:
  - `experiments/01_feature_extraction.ipynb` — extracción multi-escala. Debe producir features en el **mismo formato y rutas** que los ya guardados (idempotente: cachea y omite lo ya hecho), de modo que los features actuales en Drive sigan siendo válidos y este notebook solo se re-ejecute si hace falta.
  - `experiments/02_tsi_and_fusion.ipynb` — carga los features guardados y ejecuta todo el análisis rediseñado (TSI, matriz de ganancia, 5 estrategias de fusión, estadística) y escribe los JSON de resultados.

## Formato de los features ya extraídos (reusar, NO reextraer)

En `FEATURES_ROOT`, por dataset `ds ∈ {gtzan, fma_small, mtat, irmas}`:
- `{ds}_short.npy`, `{ds}_medium.npy`, `{ds}_long.npy`: `np.ndarray (n_tracks, 192)`, float32, **crudas / sin estandarizar**.
- `{ds}_indices.npy`: `(n_tracks,)` índices originales del dataset, ordenados; las filas de TODOS los `.npy` están alineadas por posición.
- `{ds}_labels.npy`: `(n_tracks,)` `dtype=object` (clase por track; en MTAT es multietiqueta: vector de 50 tags).
- `{ds}_splits.npy`: `(n_tracks,)` split oficial por track.
- `{ds}_errors.json`: tracks fallidos.

**Layout del vector 192-d** (crítico para el TSI por-descriptor):
`[ mean(window_means)·48 | mean(window_stds)·48 | std(window_means)·48 | std(window_stds)·48 ]`,
y dentro de cada bloque de 48, el orden de descriptores es el de `FEATURE_DIMS`:
`mfcc(20), chroma(12), spectral_centroid(1), spectral_contrast(7), spectral_rolloff(1), zcr(1), tonnetz(6)`.
El sub-vector de un descriptor a una escala = su slice en cada uno de los 4 bloques de 48 → dimensión `4×dim` (MFCC=80, ZCR=4, …). Implementa un helper `extract_descriptor(matrix_192, descriptor_name) -> (n, 4*dim)` y `descriptor_slices()` a partir de `FEATURE_DIMS`.
**Estandarización (z-score) se hace por fold, ajustada solo en train**, nunca al guardar.

## Especificación del método (resumida; el detalle pleno está en `paper/proposal.tex`)

### Ganancia de información y TSI
- `G(f,k) = 1 − L(f,k)/L_chance`, con `L(f,k)` = log-loss de un clasificador **calibrado** entrenado SOLO con el descriptor `f` a la escala `k`.
- `L_chance`: log-loss del predictor de **frecuencias de clase de entrenamiento (base rate)**, evaluado sobre el MISMO fold de evaluación que `L(f,k)`. Multiclase: `−Σ_c p_c log p_c`. Multietiqueta (MTAT): media sobre tags de la entropía binaria de prevalencia.
- **Escala óptima por selección anidada**: `k*(f)` se elige por argmax de `G` en **folds internos** y se evalúa en folds externos. El estimador puntual del TSI es la media out-of-fold de `G(f,k*_inner) − G(f,k̄)`. NO usar argmax in-sample.
- **TSI (payoff, única métrica primaria)**: `TSI(f) = [G(f,k*) − G(f,k̄)] · 1[G(f,k*) > τ]`, con baseline de convención `k̄ = 'medium'` (2 s).
- **Compuerta `τ`**: calibrada **por tarea** (el techo de G difiere entre tareas). Distribución nula por permutación de etiquetas que **re-selecciona el argmax dentro de cada réplica**; `τ` = límite superior del IC bootstrap 95% de `G` bajo la nula. Un descriptor es "temporalmente explotable" si el **límite inferior** del IC bootstrap del payoff `> 0` **y** `G(f,k*) > τ`.
- **Variante relativa (secundaria)**: `TSI_rel(f) = (G(f,k*) − G(f,k̄)) / G(f,k*)`, definida solo si la compuerta se cumple; reportar IC bootstrap; suprimir el número si el IC es demasiado ancho. Inestable cerca del piso (advertirlo). NO es la métrica principal.
- **Sesgo residual**: reportar la diferencia entre el TSI in-sample y el anidado out-of-fold como diagnóstico.
- **NO implementar** una métrica de dispersión `std_k G` (el antiguo "TSD"): se eliminó.
- **Matriz de ganancia 7×3**: `G(f,k) ± σ` por celda, con `k*(f)` y `TSI(f)` anotados. Es distinta de la matriz de importancia PI/MDI (univariada calibrada vs. multivariada); la de importancia solo corrobora, no gobierna las estrategias.

### Calibración
- XGBoost y RF: `CalibratedClassifierCV(method='sigmoid', cv=3)`. SVM: Platt (`probability=True`). MLP: temperature scaling en validación. Reportar **ECE** y reliability diagrams por clasificador. Truncar `G<0 → 0` como piso interpretativo.

### Clasificadores (4)
- **XGBoost** (principal para el TSI), **Random Forest** (500 árboles, `max_depth=30`), **SVM** (RBF, grid `C, γ` por CV 3-fold), **MLP** (2 capas ocultas 256/128, ReLU, dropout 0.3, L2 `1e-4`, Adam `lr=1e-3`, early stopping paciencia 10). Todos exponen `predict_proba`.

### Estrategias de fusión (5) — `src/fusion.py`
1. **single-scale**: la **mejor escala única**, seleccionada en folds internos (entra como UNA sola representante en la comparación estadística).
2. **early fusion**: concatenación de las 3 escalas → 576-d.
3. **late fusion (uniforme)**: promedio `1/3` de las probabilidades de los 3 clasificadores por escala.
4. **TSI-guided scale selection**: cada descriptor en su `k*(f)` (anidado), concatenado con peso unitario → 192-d.
5. **TSI-weighted late fusion**: `ŷ = Σ_k w_k p̂_k`, con `w_k` **aprendidos** en folds internos por regresión logística no negativa sobre `[p_s, p_m, p_l]` (`w_k ≥ 0`, `Σ w_k = 1`). El TSI se usa solo como **prior** de inicialización opcional `w_k ∝ ε + Σ_{f:k*(f)=k} TSI(f)`, `ε=1e-3`; reportar cuánto se aproxima el prior a los pesos aprendidos. Incluir como **referencias**: late fusion con pesos aprendidos sin prior (cota superior) y uniforme (cota inferior).

### Validación estadística — `src/stats.py`
- **Por descriptor**: Friedman omnibus sobre `G(f,s), G(f,m), G(f,l)` por fold + post-hoc de Nemenyi (responde "¿difieren las escalas?"). La etiqueta "temporalmente explotable" la gobierna el **IC del payoff** (ver arriba), no Friedman.
- **Entre estrategias de fusión**: Wilcoxon pareado por fold sobre las 5 estrategias, **Bonferroni para C(5,2)=10** (`α_corr=0.005`). Tamaño de efecto: **Cliff's delta** (umbrales 0.147/0.33/0.474). Cada comparación dentro del mismo clasificador y régimen de muestreo.
- **Consistencia**: Spearman `ρ` (IC bootstrap) entre **rankings** de TSI por dataset/clasificador (no entre valores de G). `ρ>0.7` como umbral convencional.

### Datasets y evaluación — `src/data_loader.py`, notebooks
- **GTZAN**: 5×10 CV estratificada repetida, partición fault-filtered de Kereliuk (~930). **FMA-small**: train/val/test oficial. **MTAT**: 12:1:3, multietiqueta, mAP primaria. **IRMAS**: train/test oficial; clips de 3 s ⇒ la escala larga colapsa: usar **K=2** (omitir columna larga; sustituir Friedman por Wilcoxon de rangos con signo; el TSI es direccional hacia la escala corta; la media leave-one-out degenera).

## Entregables
1. `experiments_legacy.zip` (backup) y `experiments/` recreada limpia.
2. `experiments/src/`: como mínimo `features.py` (compatible con el formato de features existente), `data_loader.py`, `classifiers.py`, `evaluation.py` (G, L_chance, ECE), `tsi.py` (TSI anidado, compuerta, TSI_rel, matriz de ganancia), `fusion.py` (5 estrategias), `stats.py` (Friedman/Nemenyi, Wilcoxon+Bonferroni, Cliff, Spearman), `importance.py` (PI/MDI, solo corroborativo), y un `results/show.py` para imprimir tablas.
3. `experiments/01_feature_extraction.ipynb` y `experiments/02_tsi_and_fusion.ipynb` con las celdas de configuración de Colab (mount, clone, paths) idénticas en estructura a las actuales.
4. Resultados a `RESULTS_ROOT` con nombres `tsi_results.json` y `experiment_results.json` (y copia a `experiments/results/`).
5. **Tests unitarios** sobre datos sintéticos pequeños (sin depender de Drive): verificar el helper de slicing por descriptor, que `G∈(−∞,1]` y se trunca a 0, que la selección anidada no usa el fold externo, que los pesos de fusión suman 1 y son ≥0, y que la compuerta y el IC del payoff se computan de forma consistente (punto, IC y τ con la misma definición de `k*`).
6. Un `experiments/README.md` corto que documente el formato de features, las rutas y cómo correr los 2 notebooks.

## Criterios de aceptación
- Ningún rastro de las fórmulas legadas (TSI=std, Wilcoxon max-min, peso multiplicativo TSI).
- `k*` se define UNA sola vez (anidado) y se usa idéntico en punto, IC, compuerta y fusión.
- El código corre end-to-end sobre datos sintéticos en los tests; los notebooks están listos para ejecutarse en Colab contra los features reales sin reextraer.
- Coincide con `paper/proposal.tex`. Ante cualquier ambigüedad, el paper manda; si el paper y este prompt difieren, detente y pregunta.
