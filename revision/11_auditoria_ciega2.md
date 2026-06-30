# Segunda auditoría CIEGA — código ↔ paper

Agente nuevo, sin seeding. **Veredicto: implementa la propuesta con discrepancias MENORES. Mayores: ninguna.** Confirmó que el weighted BCE de MTAT (MLP) ya está cableado correctamente.

## Discrepancias menores (nuevas, halladas sin sesgo)
1. **Partición de fusión FMA/MTAT (decisión paper↔código).** El notebook usa CV (KFold 5) como ruta primaria también para la **comparación de fusión**; la partición oficial se reporta como punto adicional sin Wilcoxon. Para el TSI el CV es necesario (IC por fold); para la fusión de FMA el paper implica la oficial. Fix: hacer la oficial primaria para fusión en FMA/MTAT (la maquinaria ya existe) **o** ajustar el paper para declarar CV también en fusión.
2. **Nula de τ no anidada (consistencia).** `permutation_null_kstar_gains` re-selecciona argmax sobre ganancias *outer* (no anidado), mientras el estimador puntual usa selección anidada. El argmax no anidado sobre etiquetas permutadas es más optimista → τ resulta **conservador** (error en dirección segura), pero no es "el mismo procedimiento anidado" que afirma el texto. Fix: generar la nula con el mismo esquema inner/outer, o relajar la afirmación en el paper.
3. **Submuestreo SVM más amplio que en el paper.** El paper restringe el 50% a early-fusion (R^576); el código lo aplica a toda SVM+MTAT con `len(X)>5000` (documentado en el docstring). Fix: condicionar a `input_dim==576`, o ajustar el paper a "SVM en MTAT" en general.
4. **`TSI_rel` puede exceder 1.0.** Cuando `G_baseline<0`, `payoff/G(k*)` puede superar 1 (ej. `mfcc tsi_rel=1.0226`). Fix menor: calcular `TSI_rel` con `G` truncado a `[0,1]`, o documentar que puede exceder 1 bajo calibración imperfecta.
5. **Sesgo residual no agregado.** `residual_optimism` se calcula por descriptor pero no se expone en una tabla resumen del notebook. Detalle de reporte.

## Confirmado correcto e imparcial
G=1−L/L_chance con L_chance sobre el mismo fold (verificado L=log C); selección anidada de k\* única en payoff/IC/compuerta/fusión sin winner's curse (residual_optimism>0); compuerta τ = límite superior IC nula, explotable = CI_lo>0 ∧ G(k\*)>τ; sin TSD/Wilcoxon best-vs-worst/peso multiplicativo; 5 estrategias con pesos aprendidos en simplex (SLSQP) + TSI prior + referencias; TSI-guided con k\* anidado; estandarización solo en train (sin leakage); Friedman/Nemenyi (Wilcoxon K=2), Bonferroni 10, Cliff, Spearman sobre rankings; calibración con ECE + **weighted BCE en MTAT** (`pos_weight` por tag desde train); extracción 192-d consistente con el slicing; IRMAS long colapsa; results JSON honestamente sintéticos.

## Tests
15/15 checks núcleo (numpy) PASS. Los que requieren scipy/sklearn/torch no corrieron (venv macOS incompatible + sin red); revisados estáticamente.

## Lectura
Convergiendo: cada pasada ciega encuentra issues más periféricos, ninguno bloqueante. Lo pendiente son 2 decisiones de alineación paper↔código (#1, #3), 1 consistencia que ya errа seguro (#2), y 2 pulidos (#4, #5).
