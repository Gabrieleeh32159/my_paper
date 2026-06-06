# Fase 1 — Análisis (revisión de pares objetiva)

> Insumo para la reformulación. Generado por un agente revisor independiente sobre `paper/proposal.tex` y `paper/references.bib`.

**Veredicto global (proposal):** nicho defendible (mapa diagnóstico descriptor×escala interpretable, sin deep learning) y metodología experimental cuidada. Pero arrastra un defecto estructural en su contribución central: tras quitar el peso multiplicativo, el escalar TSI ya no participa del método; el pipeline corre con G(f,k) y k*(f). El paper sigue narrando el TSI como "contribución central" y motor de la fusión, cuando la fusión la mueve `argmax_k G`. Esa disonancia es lo que un revisor marca primero.

## 1. Diagnóstico del problema del TSI

El TSI(f) = std_k G(f,k) es un escalar de dispersión que ninguna decisión del pipeline consume:
- La selección de escala usa `k*(f)=argmax_k G`, que depende de **dónde** está el máximo, no de **cuánto** se dispersa G. argmax y std son ortogonales.
- La fusión implementada concatena con peso unitario; la nota del propio paper admite que la ponderación por TSI "solo tiene efecto a nivel de decisión, no sobre features". El TSI fue removido del mecanismo por diseño.

Roles: guiar el pipeline → **No** (lo hace argmax G); guiar al practicante → necesita la celda (f,k*) y su G, no el std; habilitar generalización vía Spearman → débil (ranking de 7 puntos con std sobre K=3 es frágil); artefacto reutilizable → es la **matriz G 7×3**, no el escalar.

**std sobre K=3 es débil:** con 3 puntos std y rango son casi monótonos (la cota TSI∈[0,½·rango] lo confirma). La elección de std se lee arbitraria. Además mezcla sensibilidad temporal con capacidad discriminativa absoluta (efecto piso; MFCC ~40 dims vs ZCR ~2 dims).

**Tensión central:** dispersión ("¿importa la escala?") vs payoff accionable ("¿cuánto gano al elegir la escala correcta?"). El autor prioriza lo segundo; std no es lo segundo. Una métrica payoff (G(f,k*) − baseline) estaría conectada a k*, sería interpretable en la unidad de G, y sería la base honesta del claim de fusión.

## 2. Revisión por sección (resumen)
- **Abstract/Intro:** bien vendidos pero "utiliza el TSI para construir la fusión" ya no es cierto; resultados afirmados como hechos (sobre-venta en un proposal); nombre "TSI-weighted" contradice "TSI-scale".
- **Trabajos Relacionados + Tabla I:** lo más fuerte. Riesgos: columna "Métrica TSI" auto-cumplida; posibles mismatch cita↔contenido (Lartillot, Hamel).
- **Datasets:** Tabla II ("10-fold") vs texto ("5×10 repetido"); FMA "Oficial" vs val/test. `kereliuk2015counterexample` con título que no corresponde.
- **Extracción/agregación/fusión:** detallado y reproducible. Ec. de fusión rotulada "TSI-scale" vs nombre "TSI-weighted" en el resto. Late fusion con 1/3 fijo: lugar natural para un TSI-weighted real, hoy desaprovechado.
- **Def. TSI + G + L_chance:** G y L_chance bien definidas; log-loss normalizada es elegante. La cota refuerza std≈rango. Truncado G<0→0 censura por abajo (sesga std) y no se discute.
- **Clasificadores:** completos y honestos. XGBoost principal pero ausente del inventario de la intro.
- **Importancia:** sólido, pero **cita RQ1/RQ3 que están comentadas** (referencia huérfana). Hay **dos matrices 7×3** (G vs PI/MDI) sin distinguir.
- **Validación estadística:** lo mejor; Friedman+Nemenyi correcto. El TSI escalar ni siquiera necesita aparato estadístico propio (coherente con su irrelevancia actual).
- **Limitaciones:** buena adición; conectar el efecto-piso con el rediseño de la métrica.

## 3. Consistencia interna
- Nombre estrella inconsistente: "TSI-weighted fusion" vs "TSI-scale"/"selección de escala". El método ya no pondera.
- RQ huérfanas (línea ~435 cita RQ1/RQ3 comentadas).
- Dos matrices 7×3 sin nombrar distinto.
- Figura `methodology2.png`: verificar existencia y que la rama TSI muestre selección de escala.
- Tabla II vs texto desalineadas. Abreviaturas SC, SR sin definir.

## 4. Claims vs evidencia (suavizar)
- "Los resultados demuestran…" (3 claims como hechos) → hipótesis/objetivos.
- "reflejan propiedades intrínsecas y no artefactos" → "evidencia consistente con".
- "el TSI es la contribución central" mientras no entra al método → el más vulnerable.
- "más explicable que cualquier enfoque deep" → absoluto.
- ρ>0.7 como prueba de verdad → declarar convención.

## 5. Referencias
**Faltan en el .bib (romperán bibliografía):** `bregman1994auditory`, `fraisse1978time`, `bosch2012comparison`, `breiman2001random`, `romano2006exploring`.
**Problemas dentro del .bib:** `carvalho2023self` huérfano (nunca citado); `kereliuk2015counterexample` título no corresponde; "György" corrupto en `guinot2024semi`; `peeters2004large` debería ser techreport.

## 6. Cambios priorizados
**P0 (bloqueante — significado/accionabilidad del TSI):**
1. Redefinir TSI como payoff de la decisión: `TSI(f) = G(f,k*) − G_baseline(f)` (baseline = mean_k G o escala-convención). Significa "incertidumbre adicional resuelta al elegir bien la escala de f"; conecta con k*; base del claim de fusión. std_k G queda como estadístico secundario.
2. Devolver al TSI un rol mecánico vía **late fusion ponderada por TSI** (`ŷ=Σ_k w_k(f)p_k`), donde el peso sí decide.
3. Unificar el nombre al mecanismo real (p.ej. *TSI-guided scale selection* + *TSI-weighted late fusion*).

**P1:** añadir 5 refs faltantes; corregir kereliuk/György/carvalho; resolver RQ huérfanas; alinear Tabla II y figura.
**P2:** condicionar claims de proposal; reescribir "contribución central".
**P3:** distinguir las dos matrices 7×3; justificar comparabilidad G multiclase vs multietiqueta; definir SC/SR; XGBoost en intro; suavizar "accuracy≈F1" en IRMAS.

**Síntesis:** el paper está a una decisión de diseño de ser publicable: **qué debe significar el número TSI.** Redefinirlo como payoff de la escala óptima (P0.1) y devolverle rol mecánico vía late fusion ponderada (P0.2) resuelve a la vez accionabilidad, coherencia abstract↔método, el nombre "TSI-weighted", y la debilidad de std-sobre-K=3.
