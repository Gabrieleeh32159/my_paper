# Fase 2 — Proposición (blueprint de reformulación)

> Diseño listo para implementar **solo en el paper** (`proposal.tex`, `references.bib`). Las notas de "re-ejecutar experimentos" quedan como `% REVISAR` en el texto; **no se toca código** (decisión del autor).

## Decisión central del TSI

Se redefine el TSI de dispersión (`std_k G`) a **payoff accionable**:

```
TSI(f) = [ G(f,k*) − G(f,k̄) ] · 1[ G(f,k*) > τ ]      con k̄ = escala convención (2 s = media)
```

- **Significado concreto:** fracción de incertidumbre de etiqueta que se resuelve *de más* al extraer `f` en su escala óptima `k*` en lugar de la convención de 2 s. En la unidad de G.
- **Accionable:** dice cuánto se gana al re-escalar cada descriptor; conectado a `k* = argmax_k G`, la decisión que el pipeline sí toma. `TSI=0` ⇒ la convención ya es óptima ⇒ "no toques este descriptor".
- **Cota:** `TSI(f) ≥ 0` por construcción.
- **Compuerta de informatividad** `1[G(f,k*)>τ]`: neutraliza el efecto piso (descriptor poco informativo en todas las escalas no recibe TSI interpretable). `τ` = límite superior del IC bootstrap 95% de G bajo permutación de etiquetas.
- **Variante relativa** (comparable entre dimensionalidades): `TSI_rel(f) = (G(f,k*) − G(f,k̄)) / G(f,k*) ∈ [0,1]`.
- **`std_k G` se conserva** renombrado **TSD** (Temporal Sensitivity Dispersion) como estadístico secundario de dispersión, ya no como métrica estrella.

**Baseline elegida = convención 2 s** (no la media ni la peor escala): es el contra-fáctico honesto, porque el campo usa 1–2 s por convención (Peeters). El TSI mide exactamente lo que se gana al abandonar esa convención por descriptor.

## Rol mecánico real del TSI (nombres unificados)

| Antiguo | Nuevo | Mecanismo |
|---|---|---|
| "TSI-weighted fusion" / "TSI-scale" | **TSI-guided scale selection** | Concatena cada descriptor en su `k*`, peso unitario (interpretable). |
| (no existía) | **TSI-weighted late fusion** | `ŷ = Σ_k w_k p̂_k`, pesos derivados del TSI. Aquí el TSI **sí** decide. |

Pesos: `W_k = Σ_{f: k*(f)=k} TSI(f)`, `w_k = (ε+W_k)/Σ_j(ε+W_j)`, ε=1e-3. Caso límite TSI iguales ⇒ `w_k=1/3` (late fusion uniforme). Estimados en folds internos.

## Notas de re-ejecución (quedan como % REVISAR en el .tex; el autor corre el código aparte)
1. Nueva estrategia `TSI-weighted late fusion` (5 estrategias en vez de 4).
2. Bonferroni 6 → 10 comparaciones (`α_corr = 0.005`).
3. TSI redefinido + compuerta τ + `TSI_rel`; `std_k G` queda como TSD; IC bootstrap sobre la diferencia con selección anidada de `k*`.
4. Anotar `k*` y TSI en la matriz de ganancia; distinguirla de la matriz de importancia PI/MDI.

## Saneamiento de `references.bib`
- **Añadir 5 faltantes:** `bregman1994auditory`, `fraisse1978time`, `bosch2012comparison`, `breiman2001random`, `romano2006exploring`.
- **Corregir** `kereliuk2015counterexample` (título erróneo → "Deep Learning and Music Adversaries", IEEE TMM 17(11):2059–2071, 2015); `guinot2024semi` ("György Fazekas"); `peeters2004large` → `@techreport`.
- **Eliminar** `carvalho2023self` (huérfano, nunca citado).

## Cambios de texto (sin re-ejecutar)
Abstract y contribuciones (TSI como payoff; 5 estrategias; XGBoost en inventario; claims de proposal suavizados a hipótesis); Tabla I (columna "Métrica TSI" → "Métrica de escala"); Tabla II (GTZAN 5×10 CV rep.; FMA train/val/test); RQ huérfanas (quitar referencia a RQ1/RQ3); siglas SC/SCon/SR; dos matrices 7×3 nombradas distinto; ρ>0.7 como convención; "accuracy≈F1" en IRMAS suavizado; limitación de efecto piso conectada con compuerta + TSI_rel.

---

*El detalle LaTeX exacto de cada bloque (listo para pegar, en rojo con % REVISAR) fue generado por el agente de proposición y aplicado por el agente de revisión sobre `proposal.tex` y `references.bib`.*
