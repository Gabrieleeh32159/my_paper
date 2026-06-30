# Prompt para Claude Code — fixes tras la auditoría ciega

> Cópialo y pégalo en Claude Code abierto en la raíz del repo `my_paper`.

---

## Contexto
Una auditoría independiente del código en `experiments/` contra `paper/proposal.tex` (la **fuente de verdad**) confirmó que el método está implementado fielmente, pero dejó un punto de **protocolo** por cerrar y un par de mejoras de reporte/documentación. Arregla SOLO lo de abajo; no toques el núcleo ya correcto (TSI payoff, selección anidada de k\*, compuerta τ, L_chance, las 5 estrategias de fusión, la estadística) ni reintroduzcas fórmulas legadas. Localiza por nombre de función (las líneas pueden haberse movido).

## Arreglo 1 — (PROTOCOLO, obligatorio) Cablear *weighted binary cross-entropy* del MLP en MTAT
El paper (Sec. "Datasets" y "Modelos de clasificación") dice que en MTAT el MLP usa **weighted binary cross-entropy con pesos inversamente proporcionales a la frecuencia de cada tag**. Hoy `MLPClassifier` (en `experiments/src/classifiers.py`) **acepta** `class_weights` y lo aplica como `BCEWithLogitsLoss(pos_weight=...)`, pero **nadie calcula ni pasa esos pesos**: `make_clf_factory`/`get_classifier` no los reenvían y el notebook `02_tsi_and_fusion.ipynb` no los construye. Como resultado, MTAT corre con BCE **sin ponderar**, contradiciendo el paper.

Qué hacer:
- Calcular, **por fold y solo desde las etiquetas de entrenamiento** (sin tocar val/test), el peso por tag `pos_weight_t = n_neg_t / n_pos_t` (clipeado para evitar `inf`/`0`, p. ej. a `[1e-3, 1e3]`).
- Propagar esos pesos al `MLPClassifier` únicamente cuando `clf_name == 'mlp'` y `dataset == 'mtat'`, vía `get_classifier(...)`/`make_clf_factory` (por `**kwargs` → `class_weights`). La forma más limpia: que `make_clf_factory(dataset='mtat')` calcule los pesos dentro del `fit` a partir de `y_train`, o que la factory reciba `y_train` del fold.
- Asegurar que en las demás tareas (multiclase) el comportamiento no cambie (cross-entropy estándar).
- Añadir un test en `experiments/tests/test_method.py` que verifique que, para `dataset='mtat'` y `clf='mlp'`, el MLP recibe un `pos_weight` por tag no trivial derivado del train (y que para una tarea multiclase no se pasa).

## Arreglo 2 — (doc en el paper) Declarar que el TSI de MTAT usa CV, no la partición oficial
El paper especifica la partición oficial 12:1:3 para MTAT, pero el cómputo del **TSI** necesita múltiples folds para el IC bootstrap de la compuerta, por lo que el código usa `KFold(5)` para MTAT en el TSI (la **fusión** sí usa el split oficial). Esto es razonable pero no está dicho en el paper. Añade en `paper/proposal.tex` (Sec. de validación/datasets, en rojo con `% REVISAR`) una frase aclarando: *para MTAT, el TSI se estima sobre validación cruzada repetida (necesaria para el IC por fold de la compuerta), mientras que la comparación de fusión se reporta sobre la partición oficial 12:1:3.* No cambies resultados.

## Arreglo 3 — (reporte, opcional) Significancia de la referencia learned-LF
`fusion.py` computa `learned_lf` (pesos aprendidos sin prior, cota superior) pero el notebook solo reporta su score puntual y no lo somete a Wilcoxon vs `tsi_weighted_lf`. Si es barato, añade esa comparación pareada al bloque estadístico como referencia (sin meterla en la familia de Bonferroni de las 5 estrategias; repórtala aparte como contraste cota-superior). Si complica, déjalo documentado en el README.

## Opcional
- `MultiLabelWrapper`: para un tag degenerado (una sola clase en el train del fold) usa la prevalencia **clipeada** (coherente con `evaluation.base_rate`) en vez de `yt.mean()` cruda, para mantener consistencia con `L_chance`.

## Reglas y criterios de aceptación
- No alteres el núcleo correcto ni reintroduzcas fórmulas legadas (`std_k G`, Wilcoxon best-vs-worst, peso multiplicativo TSI).
- Mantén el workflow de Colab, los 2 notebooks y las rutas (`DRIVE_ROOT/DATA_ROOT/FEATURES_ROOT/RESULTS_ROOT`).
- Si actualizas `CLAUDE.md`, refleja en la sección "Classifiers" que el MLP usa weighted BCE en MTAT.
- Corre los tests: `python -m pytest experiments/tests/ -q` (o `python experiments/tests/test_method.py`); deben pasar, incluido el nuevo test del Arreglo 1.
- Si el paper y el código entran en conflicto, el paper manda; si te apartas del paper, actualízalo y déjalo consistente.
- Al terminar, resume qué archivos cambiaste y por qué.
