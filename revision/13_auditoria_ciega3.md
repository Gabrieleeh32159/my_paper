# Tercera auditoría CIEGA — código ↔ paper

Agente nuevo, sin seeding. **Veredicto: implementa la propuesta con discrepancias MENORES. Mayores: ninguna.**

## Confirmó los 3 fixes de la ronda anterior
- Submuestreo 50% de SVM **solo** en la representación early-fusion 576-d (`_should_subsample` verifica `dim==576`).
- `TSI_rel` acotado a [0,1] sobre G truncado.
- `residual_optimism` reportado (`show.py:show_residual_optimism`).

## Hallazgos restantes — todos paper↔código (ya cerrados en el paper)
1. **L_chance multietiqueta (forma cerrada vs evaluada):** la Ec. escribía la entropía cerrada `(1/T)ΣH(π_t)` pero el código la evalúa sobre el fold (coincide con la prosa). **Cerrado:** añadí una aclaración de que las expresiones son la forma cerrada de referencia y que operativamente L_chance se mide sobre el mismo fold (igual en expectativa).
2. **Nula del gate agregada sobre descriptores+folds:** no estaba dicho. **Cerrado:** añadí que la nula por tarea agrega réplicas sobre los 7 descriptores y los folds.
3. **MLP reserva 15% interno de validación:** no estaba dicho. **Cerrado:** añadí que, sin validación externa, el MLP reserva el 15% del train del fold para early stopping y temperature scaling.

(El punto de la nula no anidada que el auditor menciona ya estaba declarado en el paper; lo confirmó como deliberado y conservador, no un bug.)

## Confirmado correcto e imparcial
G=1−L/L_chance con truncado solo en display; selección anidada única de k\* sin fuga; TSI payoff + compuerta; TSI_rel secundaria gateada y acotada; sesgo residual; LOO; 5 estrategias con pesos aprendidos en simplex + TSI prior; calibración (XGB/RF sigmoid, SVM Platt, MLP temp scaling); MTAT (BCE ponderada por tag, submuestreo SVM solo 576-d, tags degenerados clipeados); IRMAS K=2; Friedman/Nemenyi, Wilcoxon/Bonferroni(10), Cliff, Spearman sobre rankings; layout 192-d 4×dim; sin fórmulas legadas; results JSON honestamente sintéticos.

## Tests
12 módulos compilan sin error; la suite no corrió por el venv macOS incompatible + sin red, pero `test_method.py` se revisó y cubre los puntos de riesgo (anidamiento, leakage, simplex, slices, subsample-solo-576d, BCE ponderada). Correr en Colab: `cd experiments && python -m pytest tests/ -q`.

## Lectura
Convergencia clara: tres auditorías ciegas, ninguna con discrepancias mayores ni bugs de método; los hallazgos se volvieron puramente documentales y ya están cerrados. El paper compila limpio (13 pp.).
