# Auditoría CIEGA independiente — código ↔ paper

Agente nuevo, **sin seeding**: solo se le dio el paper (fuente de verdad) + el código, sin mencionar auditorías previas ni puntos a revisar. Señal limpia.

## Veredicto
**Implementa fielmente la propuesta — discrepancias MENORES. Mayores: ninguna.**

## Hallazgo nuevo e independiente
- **M1 (protocolo):** el paper especifica *weighted binary cross-entropy* para el MLP en MTAT (pesos inversos a la frecuencia por tag). El código tiene el gancho (`MLPClassifier` aplica `class_weights` como `pos_weight`) pero **nadie computa ni pasa los pesos** (`make_clf_factory`/`get_classifier`/notebook). Tal como está, MTAT corre BCE sin ponderar → contradice el paper. **Fix:** en `make_clf_factory`, si `clf=='mlp'` y `dataset=='mtat'`, calcular `pos_weight=neg/pos` por tag desde el train del fold y propagarlo a `MLPClassifier.fit`.

## Otras menores (reporte/documentación)
- **M2:** `learned_lf` (cota superior) se computa pero no se somete a Wilcoxon vs `tsi_weighted_lf`; solo se reporta su score puntual.
- **M3:** el TSI de MTAT usa KFold(5) en vez de la partición oficial 12:1:3 (necesario para el IC por fold de la compuerta); la fusión sí usa el split oficial. Documentarlo en el paper.
- **M4:** post-hoc de Nemenyi cae a aproximación normal con Bonferroni si falta `scikit_posthocs` (afecta solo el post-hoc corroborativo, no la decisión de explotabilidad).
- **M5:** `MultiLabelWrapper` para un tag degenerado predice `yt.mean()` (0/1) en vez de la prevalencia clipeada; impacto numérico despreciable tras el clip EPS del log-loss.

## Confirmado correcto e imparcial (coincide con el paper)
G=1−L/L_chance con L_chance sobre el mismo fold; k\* anidado (inner→outer) con sesgo residual reportado y una única definición compartida en punto/IC/compuerta/fusión (sin winner's curse); TSI payoff + compuerta (explotable = IC_lo>0 ∧ G(k\*)>τ); τ por permutación que re-selecciona argmax; TSI_rel secundaria gateada; baseline LOO excluyendo k\*; 5 estrategias con pesos aprendidos en el símplex + TSI solo como prior; estadística (Friedman/Nemenyi, Wilcoxon K=2 en IRMAS, Bonferroni 10, Cliff, Spearman sobre rankings); z-score solo en train por fold (sin fuga); calibración con ECE/reliability; features 192-d con slicing 4×dim; IRMAS K=2; results JSON honestamente sintéticos.

## Tests
21/21 tests puros (numpy) PASS. 3 que requieren scipy/sklearn/torch no corrieron en el sandbox (sin red; venv macOS incompatible con Linux); revisados estáticamente, consistentes.

## Recomendación
Solo **M1** (weighted BCE en MTAT) amerita un fix de protocolo antes de los experimentos; el resto es documentación/reporte.
