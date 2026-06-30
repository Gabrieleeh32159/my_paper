# Revisión independiente del paper final corregido (main.tex, LNCS)

Agente limpio, sin seeding, como revisor de conferencia MIR sobre el paper con resultados.

## Veredicto: MAJOR REVISION
Honesto, estadísticamente cuidadoso y bien escrito. Dos bloqueantes, ambos **experimentales** (no se arreglan con texto). La diferencia entre major-revision y reject es enteramente M1.

## Fortalezas reconocidas
Rigor estadístico (CV anidada, null por permutación con re-selección de argmax, residual optimism, calibración + ECE); honestidad intelectual (no sobrevende; reconoce confounds); diseño multi-tarea que sostiene la conclusión más interesante (la sensibilidad temporal es dependiente de la tarea); resultado contraintuitivo de Tonnetz bien argumentado.

## Bloqueantes (experimentales — requieren correr, no editar)
- **M1 — Confound del número de ventanas (el decisivo).** "Corto gana para todos" puede ser artefacto: short ~150 ventanas vs long ~6 → los estadísticos across-window se estiman con muchas más muestras en corto. La caída monótona de G corto→largo en TODOS los descriptores es justo el patrón de un artefacto de estimación. **Fix: la ablación que el propio paper describe** (submuestrear ventanas de la escala corta para igualar el conteo de la larga). Barata y decisiva: si la ventaja sobrevive, el hallazgo se sostiene; si no, cambia de signo.
- **M2 — Un solo clasificador.** Se prometen RF/SVM/MLP como baselines de robustez pero quedan en "future work". La TSI depende de calibración/clasificador; el ranking debería mostrarse estable con ≥2 clasificadores (RF da MDI gratis).

## Otros mayores (parcial texto / parcial reposicionamiento)
- **M3 — Magnitud pequeña:** TSI máx ~0.085, fusión ≤0.01 F1, δ≤0.26. Reposicionar la contribución como *estudio de sensibilidad interpretable*, no como método de fusión accionable (la contribución #3 promete más de lo que entrega).
- **M4 — ECE≈0.20 en GTZAN/IRMAS:** investigar (reliability diagrams por escala / robustez al método de calibración), no solo advertir, porque GTZAN es el caso estrella y G depende de la calibración.

## Menores (text-fixable ahora)
- m1: abstract dice "2s" pero Conclusiones "1–2s" → unificar.
- m2: atenuar "ρ=1.0 highly consistent" en el abstract; preferir "rankings idénticos".
- m3: explicar el filtrado que baja IRMAS 6,705→3,756 (−44%).
- m4: referencias a "Section 3.5" inexistente (la metodología usa a/b/c/d/e) → corregir.
- m5: fricción Tabla 1 vs Tabla 3 en ZCR (ya hay footnote; considerar columna doble).
- m6: en IRMAS k*=medium para 6/7 → contradice "corto es mejor"; comentar explícito.
- m7: verificar `romano2006exploring` (venue), `fraisse1978time` (páginas), atribución de `kereliuk2015counterexample` (la partición fault-filtered suele atribuirse a Sturm).
- m8: citar librosa (`mcfee2015librosa`) en el texto.
- m9: reportar los tests de fusión de forma homogénea en las 4 tareas (o declarar dónde se hicieron).
- m10: la afirmación "learned weights ≡ TSI weights, identical performance" necesita respaldo numérico (distancia de pesos).

## Lectura
Las correcciones en rojo resolvieron lo de presentación/honestidad. Pero el verdadero techo es **M1 (ablación de conteo de ventanas)** + **M2 (2º clasificador)**: ambos requieren correr experimentos, no editar. El revisor es explícito: ejecutar la ablación de M1 es barato y decisivo.
