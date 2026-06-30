# Auditoría independiente — código `experiments/` ↔ `paper/proposal.tex`

Agente revisor independiente (ingeniero ML), leyó el paper como fuente de verdad y todo `experiments/`.

## Veredicto
**El código implementa FIELMENTE la propuesta, con discrepancias menores.** Sin fórmulas legadas (no hay `std_k G`/TSD, ni Wilcoxon best-vs-worst, ni peso multiplicativo por TSI), sin fugas de datos, sin bugs de corrección en el núcleo. Tests: 16/17 (el único fallo es `ModuleNotFoundError: sklearn` en el sandbox, no del código).

## Lo que coincide con el paper (verificado, con archivo/función)
- **TSI núcleo** (`tsi.py`): payoff `[G(k*)−G(medium)]·1[G(k*)>τ]`; `k*` por **selección anidada** (inner→outer), definición única reusada en punto/IC/compuerta/fusión; estimador out-of-fold; **sesgo optimista residual** reportado.
- **L_chance** (`evaluation.chance_log_loss`): base-rate del train evaluado sobre el mismo fold; multiclase y multietiqueta correctos; truncado `G<0→0` solo para display.
- **Compuerta τ**: percentil 97.5 de la nula por permutación que **re-selecciona argmax por réplica**, calibrada **por tarea**; "explotable" = `payoff_ci_lo>0 AND G(k*)>τ`.
- **TSI_rel**: secundaria, con IC bootstrap, solo si pasa compuerta.
- **Calibración**: XGB/RF `CalibratedClassifierCV(sigmoid,3)`, SVM Platt, MLP temperature scaling; **ECE** implementado.
- **Fusión (5)**: single-best (inner), early 576, late 1/3, TSI-guided scale selection (k\* anidado), TSI-weighted LF con **pesos aprendidos** (SLSQP no negativo, Σ=1) + TSI solo como prior; referencias learned-sin-prior y uniforme.
- **Estadística**: Friedman+Nemenyi (Wilcoxon si K=2), Wilcoxon+Bonferroni C(5,2)=10 (α=0.005), Cliff, Spearman sobre rankings con IC bootstrap.
- **Features**: slicing por descriptor en los 4 bloques de 48 (→4×dim), z-score solo en train por fold (no horneado).
- **Casos límite**: IRMAS K=2 (omite larga, Wilcoxon, TSI direccional); MTAT multietiqueta (mAP, prevalencias). Sin fuga: `k*`/`w_k` siempre desde inner folds.
- **Workflow/paths**: 2 notebooks, clone+Drive, rutas preservadas; `results/*.json` son **ejemplos sintéticos etiquetados como tales** (no se presentan como reales).

## Discrepancias menores (con fix)
- **M-4 (impacto numérico — prioridad):** en `compute_fold_gains`/`evaluate_fusion_cv` el MLP se entrena sin `X_val/y_val` → **sin early stopping ni temperature scaling** (temperatura=1.0). Fix: derivar un split interno de validación cuando `clf=='mlp'`. (No afecta a XGBoost, primario del TSI.)
- **M-6 (impacto numérico — prioridad):** `base_rate` multietiqueta no clipea prevalencias 0/1. Fix: `pi = np.clip(y_train.mean(0), eps, 1-eps)`.
- **M-3:** ECE se computa solo en escala media del primer fold; promediar sobre folds y persistir `reliability_curve` (ya existe, no se invoca).
- **M-1:** RF puede caer en backend cuML (mismos 500/30) — documentar que el RF de referencia es sklearn / qué backend se usó.
- **M-2:** registrar hiperparámetros de XGBoost (max_depth=6, n_estimators=500, lr=0.1) para reproducibilidad (el paper no los fija).
- **M-5:** FMA/MTAT/IRMAS usan CV K-fold por defecto; para la tabla de fusión final usar la **partición oficial** como indica el comentario de la celda, y documentar que el TSI usa CV repetido por necesidad estadística.

## Recomendación
Priorizar **M-4** y **M-6** (únicos con impacto numérico potencial); el resto es documentación/reproducibilidad.
