# Reauditoría independiente (post-fixes) — código ↔ propuesta

Nuevo agente auditor, desde cero, tras la última implementación.

## Veredicto
**Implementa fielmente la propuesta — discrepancias MENORES.** Núcleo correcto, sin fórmulas legadas (`std_k G`/TSD, Wilcoxon best-vs-worst, peso multiplicativo TSI), sin fugas de datos.

## Correcciones verificadas (eran los puntos con impacto numérico)
- **M-4 resuelto:** `MLPClassifier.fit` auto-provisiona validación (`train_test_split(test_size=0.15, stratify)`) cuando no recibe `X_val/y_val` → activa early stopping **y** temperature scaling dentro de TSI y fusión. (`classifiers.py` L361-374, 330-349, 460-463)
- **M-6 resuelto:** `base_rate` clipea prevalencias a `[1e-6, 1-1e-6]` (`evaluation.py` L113). Test dedicado pasa.
- **M-3 resuelto:** ECE promediado sobre folds + reliability diagram persistido (`calibration_report`, guardado en notebook cell 13).

## Confirmado coincidente con el paper
TSI payoff con k\* anidado (definición única en punto/IC/compuerta/fusión) + sesgo residual; L_chance base-rate sobre el mismo fold; compuerta τ por permutación con re-selección de argmax, por tarea; TSI_rel secundaria; calibración (XGB/RF sigmoid, SVM Platt, MLP temp scaling); 5 estrategias de fusión con pesos aprendidos + TSI prior y referencias; Friedman/Nemenyi (Wilcoxon si K=2), Wilcoxon+Bonferroni(10), Cliff, Spearman sobre rankings; slicing 4×dim y z-score solo en train; IRMAS K=2; MTAT multietiqueta; results JSON sintéticos etiquetados.

## Discrepancias menores restantes
- **M1 (única con relevancia de protocolo):** el submuestreo del 50% para SVM en MTAT/early-fusion que pide el paper no está cableado: `SVMClassifier` soporta `subsample` pero `get_classifier`/factory no lo pasan. Fix: propagar `subsample=0.5` para SVM cuando dataset==mtat, o documentar que SVM no se corre en MTAT.
- **M2 (cosmético):** default `max_depth=16` en el wrapper cuML de RF (irrelevante: se instancia con 30 explícito). Alinear default a 30.
- **M5 (cosmético):** el ejemplo sintético `irmas_example` usa 3 escalas; generarlo con `('short','medium')` para ilustrar también K=2.
- M3/M4 menores: `information_gain` devuelve 0 si `L_chance<=0` (defensivo); `N_PERM=50` para τ (subir si el tiempo lo permite).

## Tests
15/15 tests numpy-only PASS (slicing, cotas/truncado de G, L_chance, clip de prevalencias, k\* anidado inner≠outer, τ, gate/exploitable, IC del payoff, pesos en el símplex, layout TSI-guided). Los que requieren sklearn/scipy/torch no se ejecutaron por falta de dependencias en el sandbox (proxy 403); lógica revisada estáticamente; deberían pasar en Colab.

## Recomendación
Solo **M1** amerita acción antes de los experimentos (es un desvío del protocolo declarado en el paper); el resto es cosmético.
