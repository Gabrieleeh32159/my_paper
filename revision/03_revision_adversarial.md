# Fase 4 — Revisión adversarial independiente (paper final)

> Agente revisor senior, sin contexto del diseño; leyó solo `proposal.tex` + `references.bib`. Mandato: intentar rechazar.

## Veredicto: MAJOR REVISION
(tendiendo a reject si se evalúa como paper de método con resultados; defendible como proposal con revisión mayor). La reformulación cerró la disonancia previa del nombre y la scale-selection ya no es decorativa, pero tres defectos conceptuales bloquean la aceptación tal cual. Son reparables; ninguno es cosmético.

## Objeciones MAYORES (bloquean publicación)

### M1 — Sesgo de selección definido de forma inconsistente en el TSI
`TSI(f)=[G(f,k*)−G(f,k̄)]·1[G(f,k*)>τ]` con `k*=argmax`. El estimador puntual usa argmax in-sample (optimista, winner's curse), pero el IC usa selección anidada → el punto puede caer fuera de su propio IC. La compuerta hereda el sesgo: τ se fija con un IC nulo que no embebe la maximización.
**Arreglo:** definición ÚNICA y anidada de k* aplicada a punto, IC y compuerta; la distribución nula debe re-seleccionar argmax en cada permutación; reportar el sesgo optimista residual.

### M2 — `TSI_rel` inestable justo donde se la vende como solución
`TSI_rel = (G(k*)−G(k̄))/G(k*)`: el denominador es pequeño precisamente para descriptores escalares con efecto piso (ZCR), amplificando ruido; indefinida/negativa si G(k*)≤0 tras truncado. La métrica que repara el efecto piso es la más inestable bajo efecto piso.
**Arreglo:** reportar IC bootstrap; definir solo cuando el límite inferior del IC de G(k*)>τ; denominador regularizado; degradar a diagnóstico secundario con advertencia.

### M3 — Los pesos de la TSI-weighted late fusion son heurísticos y no corresponden a lo que el clasificador usa
`W_k=Σ_{f:k*(f)=k} TSI(f)`: suma TSIs univariados por descriptor para ponderar decisiones `p_k` de clasificadores **multivariados** que usan todos los descriptores. El propio paper distingue ganancia univariada vs importancia multivariada — esa distinción invalida usar TSIs univariados como pesos de decisión. Además es discontinuo (cada descriptor aporta todo su TSI a una sola escala).
**Arreglo:** justificar teóricamente, o reemplazar por pesos que midan calidad de decisión por escala (aprendidos por regresión logística sobre las p_k en folds internos, o ∝ desempeño por escala) e **incluir ese baseline fuerte** para no comparar contra hombres de paja.

## Objeciones MENORES
- **m1** Spearman multiclase vs multietiqueta: G no es conmensurable entre 10 clases y 50 tags; el ranking puede estar gobernado por la estructura de la tarea (timbre→instrumentos, armonía→género). Decir que se correlacionan *rankings* y justificar la transferencia.
- **m2** "G∈[0,1] bajo calibración perfecta" choca con la limitación 2 (G<0 truncado). Es aspiración, no garantía; todo el edificio depende de la calibración (eslabón frágil, sin resultados aún).
- **m3** La baseline k̄=2 s es ella misma una convención que el paper califica de injustificada → circularidad. Reportar sensibilidad del TSI a la elección de k̄.
- **m4** Friedman responde "¿difiere G entre escalas?", no "¿es el payoff G(k*)−G(k̄)>0?", que es lo que define el TSI. El gate de "temporalmente sensible" debería ligarse al payoff.
- **m5** Conteos: "tres clasificadores" pero se listan cuatro (XGBoost, RF, SVM, MLP); "cinco etapas" vs "cinco estrategias" confunde; verificar que no quede "6 pares / 0.0083" en tablas; precisar nivel de dimensionalidad (MFCC 20→80, ZCR 1→4 por pista).
- **m6** Fuga: k* debe estimarse en folds internos también para la scale-selection (Ec. 12), no solo para los pesos.
- **m7** Refs: `romano2006exploring` es cita de segunda mano para umbrales de Cliff (preferir Cliff 1993 / Vargha-Delaney); % REVISAR pendientes (Fraisse, Romano).
- **m8** Sobre-claims residuales: contribución 4 y "más explicable que cualquier enfoque deep" deben ir en registro hipotético; novedad descansa casi solo en "nadie más definió la métrica" (delgado para venue top).

## Para pasar a MINOR REVISION
1. k* anidado único en punto/IC/compuerta, con nula que embeba el argmax (M1).
2. Tratamiento numérico de TSI_rel + degradación a secundaria (M2).
3. Peso-TSI justificado o baseline de pesos aprendidos con comparación estadística (M3).
4. Coherencia menor (tres→cuatro clasificadores, contribuciones hipotéticas, sensibilidad a k̄).

## Novedad
Suficiente como proposal/short paper enmarcado como "métrica diagnóstica + matriz 7×3 reutilizable". Para full paper en venue competitivo es incremental sobre Bergstra/Tzanetakis salvo que los resultados superen baselines fuertes con significancia y se incluya el baseline de pesos aprendidos (M3).
