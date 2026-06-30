# Blueprint detallado — bloques LaTeX exactos para implementar

> Cada bloque indica **[REEMPLAZA: ...]** con el fragmento del `.tex` actual que sustituye. Aplicar SOLO en `proposal.tex` y `references.bib`. NO tocar código. Trabajar por coincidencia de texto (los números de línea son orientativos y pueden haberse desplazado).

---

## C.1 Abstract — [REEMPLAZA el `\begin{abstract}...\end{abstract}` completo]

```latex
% REVISAR: Abstract reescrito. Cambios: (1) el TSI se redefine como "payoff" de la
% escala optima (ganancia adicional sobre la convencion de 2 s), no como dispersion;
% (2) se distinguen las dos estrategias guiadas por TSI (seleccion de escala vs late
% fusion ponderada); (3) los claims de resultados ("demuestran que...") se suavizan a
% objetivos/hipotesis (es un proposal). Implica re-ejecutar con 5 estrategias de fusion.
\begin{abstract}
\textcolor{red}{Las \textit{audio features} artesanales como MFCCs, \textit{chroma} y descriptores espectrales siguen siendo ampliamente utilizadas en pipelines de \textit{Music Information Retrieval} (MIR), especialmente en contextos con recursos computacionales limitados donde el \textit{deep learning} no es viable. Sin embargo, la ventana temporal sobre la cual se agregan estas \textit{features} se elige típicamente por convención ---usualmente 1--2\,s o la duración completa del clip--- sin justificación acústica ni evidencia empírica comparativa. Este trabajo propone el \textbf{Índice de Sensibilidad Temporal (TSI)}, una métrica basada en ganancia de información normalizada que cuantifica \emph{cuánta incertidumbre de etiqueta adicional resuelve un descriptor al extraerse en su escala temporal óptima en lugar de la escala de convención}. A diferencia de un mero estadístico de dispersión, el TSI es un \emph{payoff} accionable, expresado en la unidad de la ganancia de información, que el practicante puede usar para decidir qué descriptores conviene re-escalar. Extraemos siete descriptores canónicos a tres resoluciones temporales ---corta ($\sim$200\,ms), media ($\sim$2\,s) y larga ($\sim$5\,s), justificadas por principios psicoacústicos--- y evaluamos cinco estrategias de fusión: \textit{single-scale}, \textit{early fusion}, \textit{late fusion}, \textit{TSI-guided scale selection} (concatenación de cada descriptor en su escala óptima) y \textit{TSI-weighted late fusion} (combinación de decisiones ponderada por el TSI). La evaluación abarca tres tareas: clasificación de género (GTZAN, FMA-small), \textit{auto-tagging} (MagnaTagATune) y reconocimiento de instrumentos (IRMAS). El TSI organiza una matriz de ganancia descriptor~$\times$~escala (7$\times$3) que sirve como referencia interpretable para practicantes de MIR y, a la vez, determina los pesos de la fusión. Nuestras hipótesis son que MFCCs y ZCR exhibirán alto TSI mientras que \textit{chroma} y \textit{tonnetz} serán temporalmente robustos, que las estrategias guiadas por TSI igualarán o superarán a la concatenación ingenua, y que estos patrones se mantendrán consistentes entre \textit{datasets} y clasificadores, ofreciendo guías prácticas para el diseño de pipelines de MIR interpretables.}
\end{abstract}
```

## C.2 Intro — [REEMPLAZA el párrafo que empieza "Este trabajo aborda esta brecha introduciendo..." y la pregunta no; mantener la pregunta central]

```latex
% REVISAR: Reescrito para alinear el rol del TSI con el mecanismo real. El TSI deja de
% "construir la fusion" en abstracto: ahora (i) cuantifica el payoff de la escala optima
% por descriptor y (ii) determina los pesos de una late fusion ponderada.
\textcolor{red}{Este trabajo aborda esta brecha proponiendo el \textbf{Índice de Sensibilidad Temporal (TSI)}, una métrica que cuantifica, para cada descriptor de audio, \emph{cuánta incertidumbre de etiqueta adicional se resuelve al extraerlo en su escala temporal óptima en lugar de la escala de convención}. El TSI no es un estadístico descriptivo de dispersión, sino un \emph{payoff} en la unidad de la ganancia de información, directamente conectado a la decisión que el pipeline toma ($k^*(f)=\arg\max_k G(f,k)$). Sobre esta métrica construimos dos estrategias: \textit{TSI-guided scale selection}, que concatena cada descriptor en su escala óptima para obtener una representación interpretable, y \textit{TSI-weighted late fusion}, que combina las decisiones de los clasificadores por escala con pesos derivados del TSI. A diferencia de trabajos previos basados en \textit{deep learning}, nuestro enfoque es interpretable: cada decisión de diseño se traza hasta una propiedad medible del descriptor.}
```

## C.3 Intro contribuciones — [REEMPLAZA ítem 1 y ítem 3 de la enumerate]

Ítem 1:
```latex
    \item \textcolor{red}{La definición y validación del \textbf{Índice de Sensibilidad Temporal (TSI)}, una métrica interpretable y accionable que cuantifica la ganancia de información adicional que cada descriptor obtiene al extraerse en su escala óptima respecto a la escala de convención, junto con una variante relativa comparable entre descriptores de distinta dimensionalidad.}
```
Ítem 3:
```latex
    \item \textcolor{red}{Una comparación sistemática de cinco estrategias de fusión (\textit{single-scale}, \textit{early fusion}, \textit{late fusion}, \textit{TSI-guided scale selection} y \textit{TSI-weighted late fusion}) en tres tareas: clasificación de género (GTZAN, FMA-small), \textit{auto-tagging} (MagnaTagATune) y reconocimiento de instrumentos (IRMAS), usando clasificadores de complejidad creciente (XGBoost, Random Forest, SVM y un MLP ligero).}
```

## C.4 Related work fusión — [REEMPLAZA el último párrafo de "Estrategias de Feature Fusion" que empieza "Ninguno de los trabajos anteriores, sin embargo, propone..."]

```latex
% REVISAR: Actualizado al nombre y mecanismo unificados. Ahora el TSI (i) selecciona
% escala por descriptor y (ii) pondera la late fusion, cerrando el ciclo diagnostico->
% mecanismo y eliminando la disonancia del nombre "TSI-weighted".
\textcolor{red}{Ninguno de los trabajos anteriores propone una estrategia de fusión explícitamente informada por la sensibilidad temporal medida de cada descriptor. Las dos estrategias guiadas por TSI que introducimos llenan este vacío: en lugar de concatenar todas las escalas con igual peso (\textit{early fusion}) o promediar decisiones independientes con peso fijo $1/3$ (\textit{late fusion}), la \textit{TSI-guided scale selection} asigna a cada descriptor su escala óptima $k^*(f)$, y la \textit{TSI-weighted late fusion} pondera la decisión de cada escala según el TSI agregado de los descriptores que la prefieren. Así, el TSI deja de ser un descriptor pasivo y se convierte en el mecanismo que decide tanto la representación como los pesos de la fusión, cerrando el ciclo entre análisis diagnóstico y diseño de pipeline.}
```

## C.5 Tabla I — [REEMPLAZA el `tabular` completo de tab:positioning]

```latex
% REVISAR: La columna "Metrica TSI" era auto-cumplida (solo este trabajo la tiene por
% definicion). Se reemplaza por "Metrica de escala" (criterio neutral: si el trabajo
% aporta UNA metrica cuantitativa de sensibilidad a la escala). No requiere re-ejecutar.
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Trabajo} & \textbf{Multi-escala} & \textbf{Interpretable} & 
\textcolor{red}{\textbf{Métrica de escala}} & \textbf{3+ tareas} \\
\midrule
Peeters \cite{peeters2004large}        & \texttimes & \checkmark & \texttimes & \texttimes \\
Bergstra \cite{bergstra2006aggregate}  & Parcial    & Parcial    & \texttimes & \texttimes \\
Lartillot \cite{lartillot2008multi}    & \checkmark & \checkmark & \texttimes & \texttimes \\
Hamel \cite{hamel2011temporal}         & Parcial    & \checkmark & \texttimes & \texttimes \\
Buisson \cite{buisson2024self}         & \checkmark & \texttimes & \texttimes & \texttimes \\
OMAR-RQ \cite{alonso2025omar}          & \texttimes & \texttimes & \texttimes & \checkmark \\
Tamm \cite{tamm2024comparative}        & \texttimes & Parcial    & \texttimes & \checkmark \\
\textbf{Este trabajo}                  & \checkmark & \checkmark & \textcolor{red}{\checkmark (TSI)} & \checkmark \\
\bottomrule
\end{tabular}
```
> Nota: preservar la fila Bergstra ya existente; si el `?` de "3+ tareas" fue fijado por el autor a `\texttimes`, respetarlo.

## C.6 Agregación/fusión — [REEMPLAZA el bloque align de las representaciones + la nota posterior]

```latex
% REVISAR: Se anaden CINCO representaciones (antes cuatro) y se renombran al esquema
% unificado. La novedad es la TSI-weighted late fusion con pesos w_k derivados del TSI,
% que devuelve al TSI un rol mecanico. Implica implementar w_k en fusion.py y re-ejecutar
% la comparacion de fusion con 5 estrategias (el autor corre el codigo aparte).
Las representaciones comparadas son \textcolor{red}{cinco}:
\begin{align}
    \text{Solo corta:}  \quad & \mathbf{x}_s \in \mathbb{R}^{192} \\
    \text{Solo media:}  \quad & \mathbf{x}_m \in \mathbb{R}^{192} \\
    \text{Solo larga:}  \quad & \mathbf{x}_l \in \mathbb{R}^{192} \\
    \text{\textit{Early fusion}:} \quad & 
        \mathbf{x}_{\text{early}} = [\mathbf{x}_s \| \mathbf{x}_m \| 
        \mathbf{x}_l] \in \mathbb{R}^{576} \\
    \text{\textit{Late fusion}:} \quad & 
        \hat{y}_{\text{late}} = \tfrac{1}{3}
        \textstyle\sum_{k} \hat{p}_k(\mathbf{x}_k) \\
    \textcolor{red}{\text{\textit{TSI-guided scale sel.}:}} \quad &
        \textcolor{red}{\mathbf{x}_{\text{TSI}} =
        [\,\mathbf{x}^{(f_1)}_{k^*(f_1)} \,\|\, \cdots \,\|\,
        \mathbf{x}^{(f_7)}_{k^*(f_7)}\,] \in \mathbb{R}^{192}} \\
    \textcolor{red}{\text{\textit{TSI-weighted late fus.}:}} \quad &
        \textcolor{red}{\hat{y}_{\text{TSI-LF}} =
        \textstyle\sum_{k} w_k\,\hat{p}_k(\mathbf{x}_k)}
\end{align}

\textcolor{red}{donde $k^*(f)= \arg\max_k G(f,k)$ es la escala óptima del descriptor $f$, $\mathbf{x}^{(f)}_{k}$ es el sub-vector del descriptor $f$ extraído a la escala $k$, y $\hat{p}_k$ es la probabilidad predicha por el clasificador entrenado en la escala $k$. Los pesos de la \textit{TSI-weighted late fusion} se derivan agregando el TSI de los descriptores que prefieren cada escala:}
\begin{equation}
\textcolor{red}{
    W_k = \!\!\!\sum_{f:\,k^*(f)=k}\!\!\! \mathrm{TSI}(f),
    \qquad
    w_k = \frac{\epsilon + W_k}{\sum_{j}(\epsilon + W_j)},
}
\label{eq:tsiweights}
\end{equation}

\textcolor{red}{con $\epsilon = 10^{-3}$ un suavizado que evita anular escalas y garantiza $w_k>0$; cuando todos los TSI son iguales se recupera $w_k=1/3$ (\textit{late fusion} uniforme) como caso límite. Los pesos $w_k$ se estiman exclusivamente en \textit{folds} internos de entrenamiento para evitar fuga de información. La \textit{TSI-guided scale selection} opera por \emph{selección} de escala con peso unitario ---evitando que la estandarización \mbox{z-score} cancele factores constantes y que los clasificadores basados en árboles, invariantes al escalado monótono, ignoren cualquier peso a nivel de \textit{feature}---; la ponderación explícita por TSI tiene efecto únicamente a nivel de decisión, en la \textit{TSI-weighted late fusion}.}
```
> Preservar la estandarización z-score que ya está descrita justo después; no duplicarla.

## C.7 Definición del TSI — [REEMPLAZA desde "El TSI es la contribución metodológica central..." hasta el párrafo de la cota natural (antes de "La calibración es indispensable...")]

```latex
% REVISAR: Nucleo del rediseno. (1) Se reformula coherente con el nuevo rol del TSI.
% (2) G y L_chance se conservan IDENTICAS. (3) El TSI se redefine como payoff
% G(f,k*)-G(f,k_conv) con compuerta de informatividad y variante relativa. (4) std_k G
% se conserva renombrado TSD como estadistico secundario. Implica (en codigo, aparte):
% compuerta tau via IC bootstrap de G(f,k*) y la variante TSI_rel.
\textcolor{red}{El TSI articula el análisis diagnóstico de este trabajo y, a la vez, determina las estrategias de fusión guiadas por él.} Para cada descriptor $f \in \{\text{MFCCs, Chroma, SC, SCon, SR, ZCR, Tonnetz}\}$ y escala $k \in \{s, m, l\}$, definimos primero la \textbf{ganancia de información normalizada}:

\begin{equation}
    G(f, k) = 1 - \frac{L(f, k)}{L_{\text{chance}}}
    \label{eq:gain}
\end{equation}

\noindent donde $L(f, k)$ es la \textit{log-loss} de un clasificador calibrado entrenado únicamente con el descriptor $f$ a la escala $k$, y $L_{\text{chance}}$ es la \textit{log-loss} del predictor constante que predice las frecuencias de clase del conjunto de entrenamiento (\textit{base rate}):

\begin{equation}
    L_{\text{chance}} = \begin{cases}
        -\displaystyle\sum_{c=1}^{C} p_c \log p_c & \text{(multiclase, } p_c \text{ frec.\ de clase)} \\[4pt]
        \dfrac{1}{T}\sum_{t=1}^{T} H(\pi_t) & \text{(multietiqueta, MTAT)}
    \end{cases}
    \label{eq:lchance}
\end{equation}

\noindent donde $p_c$ es la proporción de la clase $c$ en el entrenamiento (cuando las clases son uniformes, $p_c=1/C$, el término se reduce a $\log C$), y $H(\pi_t) = -\pi_t \log \pi_t - (1-\pi_t)\log(1-\pi_t)$ es la entropía binaria de la prevalencia $\pi_t$ del \textit{tag} $t$.

$G(f,k)$ mide qué fracción de la incertidumbre de etiqueta resuelve el descriptor $f$ a la escala $k$: vale $0$ cuando el clasificador no mejora al azar y tiende a $1$ bajo calibración perfecta. La \textbf{escala óptima} del descriptor es $k^*(f) = \arg\max_k G(f,k)$.

% REVISAR: redefinicion central del TSI.
\textcolor{red}{El \textbf{Índice de Sensibilidad Temporal} se define como la \emph{ganancia de información adicional que el descriptor obtiene al extraerse en su escala óptima en lugar de la escala de convención} $\bar{k}=m$ (2\,s), valor por defecto en la práctica de MIR \cite{peeters2004large}:}
\begin{equation}
\textcolor{red}{
    \mathrm{TSI}(f) = \bigl[\,G(f,k^*) - G(f,\bar{k})\,\bigr]\cdot
    \mathbf{1}\!\left[\,G(f,k^*) > \tau\,\right]
}
\label{eq:tsi}
\end{equation}

\textcolor{red}{El TSI se interpreta directamente en la unidad de $G$: es la fracción de incertidumbre de etiqueta que se resuelve \emph{de más} al re-escalar $f$ a su escala óptima respecto a usar la convención. Por construcción $\mathrm{TSI}(f)\geq 0$, con $\mathrm{TSI}(f)=0$ cuando la escala de convención ya es la óptima ---es decir, re-escalar ese descriptor no aporta nada y el practicante puede dejarlo en 2\,s. A mayor TSI, mayor el costo de oportunidad de adherirse a la convención. El factor indicador $\mathbf{1}[\,G(f,k^*)>\tau\,]$ es una \textbf{compuerta de informatividad} que evita el \emph{efecto piso}: un descriptor cuya ganancia óptima no supera significativamente el azar (umbral $\tau$, fijado como el límite superior del IC \textit{bootstrap} del 95\% de $G$ bajo permutación de etiquetas) recibe $\mathrm{TSI}=0$ por ser poco informativo en \emph{cualquier} escala, y su valor no se interpreta como robustez temporal. Para descriptores de dimensionalidad muy dispar reportamos además la variante relativa}
\begin{equation}
\textcolor{red}{
    \mathrm{TSI}_{\mathrm{rel}}(f) = \frac{G(f,k^*)-G(f,\bar{k})}{G(f,k^*)} \in [0,1],
}
\label{eq:tsirel}
\end{equation}
\textcolor{red}{que expresa el payoff como fracción de la discriminabilidad máxima del descriptor y es comparable entre descriptores escalares (ZCR, $\sim$2 dims) y de alta dimensión (MFCC, $\sim$40 dims). Reportamos $\mathrm{TSI}$ (absoluto) como métrica primaria accionable y $\mathrm{TSI}_{\mathrm{rel}}$ como complemento para comparaciones cruzadas.}

% REVISAR: std_k G se conserva pero degradado a estadistico secundario y renombrado TSD.
\textcolor{red}{Como estadístico secundario de \emph{dispersión} ---que responde a la pregunta descriptiva ``¿varía la discriminabilidad de $f$ entre escalas?'' pero no a ``¿cuánto se gana al elegir bien la escala?''--- reportamos la \textbf{Dispersión de Sensibilidad Temporal} $\mathrm{TSD}(f)=\sqrt{\tfrac{1}{K}\sum_k (G(f,k)-\bar G(f))^2}$, con $\bar G(f)=\tfrac1K\sum_k G(f,k)$. El TSD contextualiza el perfil temporal del descriptor, pero las decisiones de fusión y la guía al practicante se basan en el TSI.}
```
> El párrafo "La calibración es indispensable..." se conserva tal cual a continuación.

## C.8 Validación por descriptor — [REEMPLAZA el párrafo de la sección TSI que describe la validación estadística por descriptor (el que ya quedó con Friedman en rojo)]

Añadir al final de ese párrafo (conservando lo de Friedman/Nemenyi ya presente) la parte del IC del TSI redefinido:
```latex
% REVISAR: Se conserva Friedman+Nemenyi. Se anade que el IC bootstrap ahora se computa
% sobre la diferencia G(f,k*)-G(f,k_conv) con seleccion anidada de k*, y que la compuerta
% de informatividad usa el mismo aparato.
\textcolor{red}{La incertidumbre del TSI redefinido (Ec.~\ref{eq:tsi}) se cuantifica con un intervalo de confianza \textit{bootstrap} del 95\% (1\,000 remuestreos sobre folds) sobre la diferencia $G(f,k^*)-G(f,\bar k)$; como $k^*$ se selecciona por \textit{argmax} sobre los mismos datos, el IC se computa con selección anidada de $k^*$ en \textit{folds} internos para evitar sesgo optimista, y la compuerta de informatividad se activa cuando el límite inferior del IC de $G(f,k^*)$ supera $\tau$.}
```

## C.9 Matriz de ganancia — [REEMPLAZA el párrafo "Los resultados se visualizan como una matriz de sensibilidad 7×3..."]

```latex
% REVISAR: Se nombra explicitamente "matriz de ganancia G" para distinguirla de la matriz
% de importancia PI/MDI (tambien 7x3).
\textcolor{red}{Los resultados se visualizan como una \textbf{matriz de ganancia $7\times 3$} (descriptores $\times$ escalas), donde cada celda contiene $G(f,k)\pm\sigma$, sobre la cual se anotan $k^*(f)$ y el TSI por descriptor. Esta matriz de ganancia es conceptualmente distinta de la \emph{matriz de importancia} $7\times 3$ basada en \textit{permutation importance} y MDI (Sec.~\ref{sec:importancia}): la primera mide discriminabilidad univariada calibrada por descriptor aislado; la segunda mide contribución marginal dentro del modelo multivariado conjunto.}
```

## C.10 Importancia / RQ huérfanas — [REEMPLAZA el encabezado de la subsección de importancia y su frase introductoria con RQ1/RQ3]

```latex
% REVISAR: Se eliminan las referencias huerfanas a RQ1/RQ3 (las RQ estan comentadas). Se
% anade \label{sec:importancia} para el cruce con la matriz de ganancia.
\subsection{Análisis de importancia de \textit{features}}
\label{sec:importancia}

\textcolor{red}{Para complementar el análisis univariado del TSI con una medida de contribución dentro del modelo conjunto, analizamos la importancia de \textit{features} usando tres métodos complementarios, ordenados de mayor a menor confiabilidad estadística:}
```

## C.11 Bonferroni 6→10 — [REEMPLAZA el párrafo "Entre estrategias de fusión:"]

```latex
% REVISAR: Cinco estrategias => C(5,2)=10 comparaciones => Bonferroni de 6 a 10
% (alpha_corr=0.005). Implica re-ejecutar el bloque de comparacion de fusion.
\textbf{Entre estrategias de fusión:} \textcolor{red}{Las comparaciones entre las cinco estrategias (\textit{single-scale}, \textit{early}, \textit{late}, \textit{TSI-guided scale selection} y \textit{TSI-weighted late fusion}) se evalúan con la prueba de Wilcoxon sobre las métricas por fold, con corrección de Bonferroni para las $\binom{5}{2}=10$ comparaciones por pares ($\alpha_{\text{corr}}=0.05/10=0.005$).} Los tamaños de efecto se reportan con delta de Cliff: pequeño $|\delta|\geq 0.147$, mediano $|\delta|\geq 0.33$, grande $|\delta|\geq 0.474$ \cite{romano2006exploring}.
```

## C.12 Spearman ρ>0.7 — [REEMPLAZA el párrafo final de consistencia Spearman]

```latex
% REVISAR: rho>0.7 se declara convencion (no prueba); "propiedades intrinsecas" -> 
% "evidencia compatible con".
\textcolor{red}{La consistencia de los rankings de TSI entre tareas y clasificadores se evalúa con el coeficiente de correlación de Spearman $\rho$ (IC 95\% \textit{bootstrap}). Adoptamos $\rho>0.7$ como umbral convencional para considerar los rankings consistentes; una consistencia alta se interpreta como \emph{evidencia compatible con} que los patrones reflejan propiedades acústicas de los descriptores más que artefactos del clasificador o del \textit{dataset}, sin constituir prueba definitiva.}
```

## C.13 Limitaciones efecto piso — [REEMPLAZA la primera oración(es) del bloque de Limitaciones sobre dimensionalidad/efecto piso, conservando calibración e IRMAS]

```latex
% REVISAR: La limitacion de dimensionalidad/efecto piso se conecta con el rediseno: la
% compuerta y TSI_rel mitigan ambos confundidos.
\textcolor{red}{Tres limitaciones acotan la interpretación del TSI. Primero, comparar la ganancia $G$ entre descriptores de dimensionalidad muy dispar es un confundido: tras la agregación media+desviación, el MFCC aporta $\sim$40 dimensiones frente a $\sim$2 del ZCR, de modo que un payoff temporal pequeño en descriptores escalares podría reflejar un \emph{efecto piso} ---poca capacidad discriminativa absoluta en cualquier escala--- y no robustez temporal genuina. Mitigamos este confundido por dos vías: la compuerta de informatividad (Ec.~\ref{eq:tsi}) anula el TSI de descriptores cuya ganancia óptima no supera el azar, y la variante relativa $\mathrm{TSI}_{\mathrm{rel}}$ (Ec.~\ref{eq:tsirel}) normaliza el payoff por la ganancia máxima del propio descriptor, haciéndolo comparable entre dimensionalidades. Aun así, las comparaciones absolutas de TSI son más fiables entre descriptores de dimensionalidad comparable.}
```
> Conservar el resto del bloque de Limitaciones (calibración/ECE/G<0 e IRMAS) sin cambios.

---

# D — references.bib

## D.1 Añadir 5 entradas
```bibtex
@book{bregman1994auditory,
  title     = {Auditory Scene Analysis: The Perceptual Organization of Sound},
  author    = {Bregman, Albert S.},
  year      = {1994},
  publisher = {MIT Press},
  address   = {Cambridge, MA},
  note      = {Paperback edition; first published 1990}
}

@incollection{fraisse1978time,
  title     = {Time and Rhythm Perception},
  author    = {Fraisse, Paul},
  booktitle = {Handbook of Perception, Vol.\ VIII: Perceptual Coding},
  editor    = {Carterette, Edward C. and Friedman, Morton P.},
  pages     = {203--254},
  year      = {1978},
  publisher = {Academic Press},
  address   = {New York}
}
% REVISAR: rango de paginas de Fraisse aproximado; confirmar con el volumen impreso.

@inproceedings{bosch2012comparison,
  title     = {A Comparison of Sound Segregation Techniques for Predominant Instrument Recognition in Musical Audio Signals},
  author    = {Bosch, Juan J. and Janer, Jordi and Fuhrmann, Ferdinand and Herrera, Perfecto},
  booktitle = {Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR)},
  pages     = {559--564},
  year      = {2012},
  address   = {Porto, Portugal}
}

@article{breiman2001random,
  title     = {Random Forests},
  author    = {Breiman, Leo},
  journal   = {Machine Learning},
  volume    = {45},
  number    = {1},
  pages     = {5--32},
  year      = {2001},
  publisher = {Springer},
  doi       = {10.1023/A:1010933404324}
}

@inproceedings{romano2006exploring,
  title     = {Exploring Methods for Evaluating Group Differences on the {NSSE} and Other Surveys: Are the t-test and Cohen's d Indices the Most Appropriate Choices?},
  author    = {Romano, Jeanine and Kromrey, Jeffrey D. and Coraggio, Jesse and Skowronek, Jeff},
  booktitle = {Annual Meeting of the Southern Association for Institutional Research},
  year      = {2006},
  address   = {Arlington, VA}
}
% REVISAR: venue de Romano et al. por confirmar (Southern vs Florida Association for Institutional Research).
```

## D.2 Corregir kereliuk
```bibtex
% REVISAR: titulo previo ("piano transcription") no correspondia. La cita apunta a la
% particion fault-filtered de GTZAN, de Kereliuk, Sturm & Larsen, IEEE TMM (verificado).
% Se conserva la clave para no romper citas.
@article{kereliuk2015counterexample,
  title     = {Deep Learning and Music Adversaries},
  author    = {Kereliuk, Corey and Sturm, Bob L. and Larsen, Jan},
  journal   = {IEEE Transactions on Multimedia},
  volume    = {17},
  number    = {11},
  pages     = {2059--2071},
  year      = {2015},
  publisher = {IEEE},
  doi       = {10.1109/TMM.2015.2478068}
}
```

## D.3 Corregir guinot (György)
```bibtex
@article{guinot2024semi,
  title   = {Semi-Supervised Contrastive Learning of Musical Representations},
  author  = {Guinot, Julien and Quinton, Elio and Fazekas, Gy{\"o}rgy},
  journal = {arXiv preprint arXiv:2407.13840},
  year    = {2024}
}
```

## D.4 Eliminar carvalho2023self (huérfano, nunca citado).

## D.5 peeters → techreport
```bibtex
@techreport{peeters2004large,
  title       = {A Large Set of Audio Features for Sound Description (Similarity and Classification) in the {CUIDADO} Project},
  author      = {Peeters, Geoffroy},
  institution = {IRCAM},
  number      = {CUIDADO I.S.T. Project Report},
  year        = {2004},
  pages       = {1--25}
}
```

## D.6 Tabla II — alinear filas GTZAN y FMA
```latex
GTZAN \cite{tzanetakis2002musical} & Clasif.\ género & 1\,000 & 
    10 géneros & \textcolor{red}{$5{\times}10$ CV rep.} \\
FMA-small \cite{defferrard2017fma} & Clasif.\ género & 8\,000 & 
    8 géneros  & \textcolor{red}{Oficial (tr/val/test)} \\
```

## D.7 Siglas SC/SCon/SR en la lista de descriptores
- Spectral Centroid → `(\textcolor{red}{SC}, 1 valor)`
- Spectral Contrast → `(\textcolor{red}{SCon}, 7 bandas)`
- Spectral Rolloff → `(\textcolor{red}{SR}, 1 valor)`

## D.8 IRMAS accuracy≈F1 — [REEMPLAZA el ítem IRMAS de métricas]
```latex
    \item \textbf{Reconocimiento de instrumentos} (IRMAS): \textit{Accuracy} global y F1 macro-promediado. \textcolor{red}{Como la distribución de clases es aproximadamente balanceada, ambas métricas tienden a ser cercanas; reportamos las dos para facilitar la comparación con la literatura y para detectar desbalances residuales por clase.}
```
