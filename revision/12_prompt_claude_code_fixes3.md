# Prompt para Claude Code — fixes tras la 2ª auditoría ciega (solo código)

> Cópialo y pégalo en Claude Code abierto en la raíz del repo `my_paper`.
> Nota: los cambios al paper (`paper/proposal.tex`) ya se hicieron por separado; aquí NO toques `paper/`. Solo `experiments/` y, si aplica, la sección "Classifiers" de `CLAUDE.md`.

## Contexto
Una auditoría independiente del código en `experiments/` contra `paper/proposal.tex` (la **fuente de verdad**) confirmó que el método está implementado fielmente. Quedan tres ajustes menores. No toques el núcleo ya correcto (TSI payoff, selección anidada de k\*, compuerta τ, L_chance, las 5 estrategias de fusión, la estadística) ni reintroduzcas fórmulas legadas (`std_k G`/TSD, Wilcoxon best-vs-worst, peso multiplicativo TSI). Localiza por nombre de función.

## Arreglo 1 — (PROTOCOLO, obligatorio) Restringir el submuestreo 50% de SVM a early-fusion
El paper restringe el submuestreo aleatorio del 50% a la **representación early-fusion ($\mathbb{R}^{576}$)** de SVM en MTAT. Hoy el código lo aplica a **toda** SVM+MTAT con `len(X)>5000` (en `make_clf_factory`/`SVMClassifier`, ver el `subsample` con umbral fijo), lo que incluye los ajustes single-descriptor del TSI — una desviación del paper.

Qué hacer:
- Condicionar el submuestreo del 50% a `input_dim == 576` (early fusion) además de `dataset=='mtat'` y `clf=='svm'`. Los ajustes de menor dimensión (single-descriptor del TSI, single-scale 192-d) usan el train completo.
- Mantén la semilla fija y que el submuestreo sea solo sobre el train del fold.
- Actualiza el docstring de `make_clf_factory`/`SVMClassifier` para reflejar "solo early-fusion 576-d", y alinea la línea correspondiente de la sección **Classifiers** de `CLAUDE.md` (que debe decir: submuestreo 50% solo en la representación early-fusion de SVM en MTAT).
- Añade/ajusta un test que verifique que SVM+MTAT con `input_dim==576` submuestrea y con `input_dim<576` no.

## Arreglo 2 — (corrección menor) `TSI_rel` acotado a [0,1]
En `src/tsi.py` (`tsi_from_fold_gains`), `tsi_rel = payoff / G(k*)` puede exceder 1.0 cuando `G_baseline < 0` (calibración imperfecta), p. ej. `mfcc tsi_rel=1.0226` en los JSON de ejemplo. El paper define `TSI_rel` como fracción de la discriminabilidad máxima (∈ [0,1]).

Qué hacer:
- Calcular `TSI_rel` usando `G` truncado al piso 0 en numerador y denominador (es decir, con `G⁺ = max(G,0)`): `TSI_rel = (G⁺(k*) − G⁺(k̄)) / G⁺(k*)`, y además recortar el resultado a `[0,1]` por seguridad numérica.
- Documentar en el docstring que, bajo calibración imperfecta, `TSI_rel` se reporta sobre `G` truncado y por eso queda en `[0,1]`. No cambies el `TSI` absoluto (que sí usa `G` crudo).
- Regenera los JSON de ejemplo (`results/_gen_example_results.py`) si exponen `tsi_rel>1`, para que el ejemplo sea coherente.

## Arreglo 3 — (reporte) Exponer el sesgo optimista residual en el resumen
`residual_optimism` ya se computa por descriptor en `tsi.py`, pero no se muestra en una tabla resumen del notebook. En `02_tsi_and_fusion.ipynb` (celda de render / `results/show.py`), añade una columna o una pequeña tabla que reporte, por dataset/clasificador, el TSI in-sample, el TSI anidado y su diferencia (`residual_optimism`), como diagnóstico de honestidad del estimador que el paper pide reportar.

## Reglas y criterios de aceptación
- No alteres el núcleo correcto ni reintroduzcas fórmulas legadas.
- Mantén el workflow de Colab, los 2 notebooks y las rutas (`DRIVE_ROOT/DATA_ROOT/FEATURES_ROOT/RESULTS_ROOT`). NO edites `paper/`.
- Corre los tests: `python -m pytest experiments/tests/ -q` (o `python experiments/tests/test_method.py`); deben pasar, incluido el nuevo test del Arreglo 1.
- Al terminar, resume qué archivos cambiaste y por qué.
