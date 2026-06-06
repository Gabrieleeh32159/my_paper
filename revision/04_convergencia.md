# Bucle de revisión iterativa — registro de convergencia

Cada revisor fue un agente **independiente nuevo**, lanzado desde cero, que leyó únicamente `proposal.tex` y `references.bib` (sin contexto de revisores previos ni de las notas de diseño).

| Iteración | Revisor (fresco) | Veredicto | Bloqueantes |
|---|---|---|---|
| 1 | Adversarial (modo paper de método) | MAJOR REVISION | M1 sesgo de selección, M2 sin resultados, M3 pesos heurísticos + menores |
| — | *(arreglos: k\* anidado único, baseline canónica, colapso del zoo de métricas, pesos aprendidos + prior honesto, etc.)* | | |
| 2 | Adversarial (modo paper de método) | MAJOR REVISION | M1 TSI no mecánico, **M2 sin resultados**, M3 baseline circular, M4 zoo de métricas, M5 confundido cross-task |
| — | *(arreglos conceptuales M1/M3/M4/M5 + menores; reencuadre explícito como **propuesta**)* | | |
| 2′ | **Modo propuesta** (no penaliza falta de resultados) | **PUBLICABLE COMO PROPUESTA** (minor revision) | Ninguno. 3 menores-mayores de redacción |
| — | *(arreglos: conteo Bonferroni, caso degenerado del prior, media leave-one-out, Friedman→Wilcoxon en IRMAS)* | | |
| 3 | **Modo propuesta** (confirmación, nuevo desde cero) | **PUBLICABLE COMO PROPUESTA** (minor revision) | **"Ninguna que impida publicar como propuesta."** |
| — | *(arreglos finales: coherencia Sec. importancia, τ por tarea, IRMAS direccional)* | | |

## Veredicto final
Dos revisores independientes consecutivos, cada uno desde cero, dictaminaron **publicable como propuesta** sin objeciones bloqueantes. El revisor final destacó que el tratamiento del sesgo de selección es "rigor de nivel publicable, raramente visto incluso en papers de resultados" y que la propuesta "anticipa y neutraliza, en el diseño, las objeciones que normalmente hundirían un trabajo de este tipo".

## El cambio que lo desbloqueó
El giro decisivo fue redefinir el TSI de **dispersión** (`std_k G`, desconectada del método) a **payoff accionable** (`G(f,k*) − G(f,k̄)`): un número con significado concreto (incertidumbre adicional resuelta al elegir la escala óptima vs. la convención), conectado a la decisión real del pipeline (k\*), y honestamente caracterizado (diagnóstico que motiva la representación y sirve de prior, no un peso mágico).

## Pendientes menores para cámara lista (no bloqueantes)
- Limpiar el texto en rojo y los comentarios `% REVISAR` una vez aprobados los cambios.
- Confirmar 2 metadatos bibliográficos (`fraisse1978time` páginas; `romano2006exploring` venue).
- Regenerar `methodology2.png` para reflejar las 5 estrategias y las dos matrices 7×3 distintas.
- Renombrado opcional "TSI-weighted late fusion" → "late fusion con prior TSI" (el revisor lo sugirió; el TSI inicializa, no pondera).
- **Ejecutar el estudio**: el método y protocolo están pre-registrables; al obtener resultados, volver a someter a revisión evaluando ya los números.
