# Prompt para Claude Code — arreglos pendientes (post-auditoría)

> Cópialo y pégalo en Claude Code abierto en la raíz del repo `my_paper`.

---

## Contexto
Una auditoría independiente del código en `experiments/` contra `paper/proposal.tex` (la **fuente de verdad**) confirmó que el método está implementado fielmente, pero dejó algunas discrepancias por cerrar. Arregla SOLO lo de abajo, sin alterar el método ya correcto y sin reintroducir fórmulas legadas (`std_k G`/TSD, Wilcoxon best-vs-worst, peso multiplicativo por TSI). Localiza cada punto por nombre de función (las líneas pueden haberse movido).

## Arreglo 1 — (PROTOCOLO, obligatorio) Cablear el submuestreo 50% de SVM en MTAT
El paper (Sec. "Modelos de clasificación") dice: *para la representación early fusion ($\mathbb{R}^{576}$), el entrenamiento del SVM se realiza con submuestreo aleatorio del 50% de las pistas de entrenamiento en MTAT*. Hoy `SVMClassifier` (en `experiments/src/classifiers.py`) acepta un parámetro `subsample`, pero `get_classifier()` y la factory `make_clf_factory` del notebook `02_tsi_and_fusion.ipynb` **nunca lo pasan**, así que queda inerte.

Qué hacer:
- Propagar `subsample=0.5` al `SVMClassifier` únicamente cuando `clf_name == 'svm'` y `dataset == 'mtat'` (idealmente solo para la representación early-fusion 576-d; si esa distinción complica la factory, aplicarlo a SVM+MTAT en general y documentarlo).
- Asegurar que el submuestreo se haga **solo sobre el conjunto de entrenamiento del fold** (nunca sobre validación/test) y con semilla fija para reproducibilidad.
- Que `get_classifier(...)` acepte y reenvíe `subsample` por `**kwargs` hasta `SVMClassifier`.
- Si decides en cambio NO correr SVM en MTAT, entonces actualiza el paper (`paper/proposal.tex`) para que diga eso explícitamente; pero la opción preferida es cablear el 50%.

## Arreglo 2 — (cosmético) Default engañoso de profundidad en el wrapper RF de cuML
En `classifiers.py`, el `__init__` del wrapper de cuML (`_CuMLRFWrapper`) tiene `max_depth=16` por defecto, aunque en la práctica `_make_rf_estimator` siempre instancia con `max_depth=30`. Alinea el default del wrapper a `max_depth=30` para evitar confusión. (RF de referencia del paper: 500 árboles, profundidad 30.)

## Arreglo 3 — (cosmético) Ejemplo sintético de IRMAS debe ilustrar K=2
En `experiments/results/_gen_example_results.py`, el bloque `irmas_example` se genera con 3 escalas; el paper marca IRMAS como **K=2** (la escala larga colapsa). Genera ese ejemplo con `scales=('short','medium')` para que el JSON de ejemplo también refleje el camino K=2 (matriz de ganancia sin columna larga, Friedman→Wilcoxon). Mantén el etiquetado de "ejemplo sintético / no es un run real".

## Arreglo 4 — (reproducibilidad, doc) Registrar hiperparámetros de XGBoost
El paper fija RF (500/30) y la calibración, pero no los hiperparámetros de XGBoost que usa el código (p. ej. `max_depth`, `n_estimators`, `learning_rate`). Añádelos a `CLAUDE.md` (sección de clasificadores) y, si corresponde, a un breve `config` o al texto del paper, para que el estudio sea reproducible. No cambies los valores; solo documéntalos tal como están en `classifiers.py`.

## Opcionales (no bloqueantes)
- `evaluation.information_gain`: hoy devuelve `0.0` si `L_chance<=0`. Está bien como defensa, pero añade un `warnings.warn` para no enmascarar silenciosamente ese caso degenerado.
- `N_PERM` para la nula de τ: el notebook usa 50; deja un comentario o parámetro para subirlo (p. ej. 200) cuando el tiempo de cómputo lo permita.

## Reglas y criterios de aceptación
- No toques el núcleo ya correcto (TSI payoff, selección anidada de k\*, compuerta τ, L_chance, las 5 estrategias de fusión, la estadística). No reintroduzcas fórmulas legadas.
- Mantén el workflow de Colab, los 2 notebooks y las rutas (`DRIVE_ROOT/DATA_ROOT/FEATURES_ROOT/RESULTS_ROOT`).
- Corre los tests: `python -m pytest experiments/tests/ -q` (o `python experiments/tests/test_method.py`). Deben seguir pasando; añade un test que verifique que SVM+MTAT recibe `subsample=0.5` a través de la factory.
- Si algo del paper y el código entran en conflicto al cablear el Arreglo 1, **el paper manda**; si decides apartarte del paper, actualízalo en `proposal.tex` y déjalo consistente.
- Al terminar, resume qué archivos cambiaste y por qué.
