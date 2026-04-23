# Abstract

**Problema:** La Recuperación de Información Musical (MIR, por sus siglas en inglés) ha logrado avances significativos utilizando modelos de aprendizaje profundo para extraer representaciones de audio. Sin embargo, la mayoría de los enfoques actuales extraen características a corto plazo (patrones acústicos locales) o colapsan la dimensión temporal en representaciones planas, perdiendo la intrincada estructura temporal y jerárquica de la música (donde notas forman motivos, motivos forman frases y frases forman secciones). 
**Enfoque propuesto:** Este proyecto propone *Beyond Local Acoustic Patterns*, un modelo basado en un libro de códigos jerárquico y multiescala (Hierarchical Multi-Scale Codebook). Se busca ir más allá de las características locales mediante la creación de un espacio latente discreto que cuantice y capture la información musical en múltiples niveles de granularidad temporal de manera simultánea.
**Metodología:** Se implementará un enfoque de aprendizaje autosupervisado (Self-Supervised Learning, SSL). Utilizaremos una arquitectura de codificador profundo acoplada a múltiples capas de cuantización vectorial (codebooks), optimizada a través de predicción de tokens enmascarados (Masked Token Prediction) y aprendizaje contrastivo en distintos niveles temporales. Evaluaremos el modelo en conjuntos de datos estándar (MagnaTagATune, FMA, SALAMI, OpenMIC) a través de tareas <i>downstream</i> como auto-etiquetado, segmentación estructural y clasificación de instrumentos.
**Potencial impacto:** Proveerá una representación compacta, altamente eficiente para indexación y búsqueda, que preserva la comprensión semántica a largo y corto plazo de las pistas musicales. Esto beneficiará directamente tanto la investigación fundamental en MIR como las aplicaciones industriales en sistemas de recomendación y navegación intra-pista.

***

# Introduction

**Contexto del problema:** En la última década, el uso de representaciones de audio preentrenadas y técnicas de aprendizaje de transferencia (transfer learning) ha dominado el campo de MIR. Las arquitecturas buscan extraer embeddings robustos que sirvan para diversas tareas, minimizando la necesidad de grandes conjuntos de datos etiquetados.
**Motivación:** La música es inherentemente jerárquica. Las soluciones actuales logran comprimir el audio y extraer conocimiento local, pero a menudo fallan al capturar dependencias temporales de gran escala de manera estructurada. Por otro lado, la discretización mediante libros de códigos (Vector Quantization/Codebooks) ha demostrado ser crucial para la eficiencia (bajo almacenamiento y búsqueda rápida) y compresión en modelos recientes.
**Gap en la literatura:** La literatura reciente se divide en dos: modelos que exploran libros de códigos y predicción de tokens a una resolución temporal única plana, y modelos que exploran representaciones continuas jerárquicas sin los beneficios de compresión e indexación que ofrece la cuantización. Falta un modelo que unifique **representaciones discretas (codebooks)** con una **arquitectura explícitamente jerárquica y multiescala**.
**Objetivo de la investigación:** Diseñar, entrenar y evaluar un modelo de aprendizaje autosupervisado que genere un libro de códigos jerárquico multiescala, capaz de representar tanto micro-patrones tímbricos como macro-estructuras funcionales de una pista musical.
**Contribuciones esperadas:** 
1. Una nueva arquitectura de cuantización jerárquica.
2. Un modelo preentrenado de código abierto útil para múltiples tareas <i>downstream</i>.
3. Un marco de evaluación empírico sobre cómo las diferentes escalas del codebook impactan tareas locales (ej. detección de tono) vs. globales (ej. estructura/etiquetado).

***

# Related Work

**1. Alonso-Jiménez et al. (2025) - *OMAR-RQ: Open Music Audio Representation Model Trained with Multi-Feature Masked Token Prediction***
*   **Descripción:** Propone un modelo autosupervisado basado en predicción de tokens enmascarados combinando múltiples características de entrada (mel, CQT, EnCodec) y esquemas de cuantización escalar finita y codebooks paralelos.
*   **Comparación crítica:** OMAR-RQ demuestra que el uso de múltiples <i>codebooks</i> mejora el rendimiento y enriquece el espacio objetivo. Sin embargo, **su limitación actual** es que opera en una escala temporal plana; no desentraña la jerarquía de los eventos musicales a distintos niveles, limitando su rendimiento en tareas fuertemente dependientes del contexto a largo plazo.

**2. Buisson et al. (2024) - *Self-Supervised Learning of Multi-level Audio Representations for Music Segmentation***
*   **Descripción:** Aborda la segmentación estructural utilizando aprendizaje contrastivo para aprender representaciones continuas multinivel que operan a diferentes escalas temporales simultáneamente.
*   **Comparación crítica:** Excelente para desentrañar jerarquías musicales (nivel grueso vs. refinado) mediante máscaras en los embeddings. La **limitación** principal es que sus embeddings son continuos; carece de la eficiencia computacional, la capacidad de indexación rápida y la reducción de ruido semántico que un enfoque basado en *codebooks* discretos proporcionaría.

**3. Guinot et al. (2024) - *Semi-supervised contrastive learning of musical representations***
*   **Descripción:** Introduce SemiSupCon, un enfoque que combina objetivos contrastivos supervisados y autosupervisados para inyectar conocimiento del dominio musical en el espacio latente utilizando datos parcialmente etiquetados.
*   **Comparación crítica:** Mejora considerablemente la robustez y calidad de la métrica de similitud en el espacio latente. Su **limitación** radica en que la métrica de similitud se modela a través de una arquitectura estándar que colapsa la información de la pista, perdiendo la resolución multiescala necesaria para segmentación fina o análisis de la estructura general simultáneamente.

**4. Tamm & Aljanaki (2024) - *Comparative Analysis of Pretrained Audio Representations in Music Recommender Systems***
*   **Descripción:** Evalúa la aplicabilidad de seis modelos <i>backend</i> preentrenados (como MERT, EncodecMAE, MusiCNN) en sistemas de recomendación musical híbridos.
*   **Comparación crítica:** El estudio demuestra una alta variabilidad: los modelos que sobresalen en tareas de MIR clásicas (auto-etiquetado, análisis estructural) no siempre rinden bien en recomendaciones. La **limitación actual** de la representación es la rigidez del modelo <i>backend</i>; una representación jerárquica multiescala (nuestra propuesta) podría permitir que un sistema de recomendación extraiga dinámicamente el nivel de escala (ej. tímbrica local vs. estructural global) más pertinente para cada usuario.

**5. Ding & Lerch (2023) - *Audio embeddings as teachers for music classification***
*   **Descripción:** Combina la destilación de conocimiento y el aprendizaje por transferencia, utilizando embeddings de audio de alta complejidad como "profesores" para regularizar redes "estudiantes" de baja complejidad en tareas de clasificación.
*   **Comparación crítica:** Prueba que la transferencia de conocimiento latente es extremadamente efectiva. No obstante, **la limitación** es que la calidad temporal de la destilación está estrangulada por la resolución fija y a menudo colapsada del modelo profesor (ej. VGGish o OpenL3 con resolución plana). Nuestro enfoque jerárquico superaría esto proporcionando un "profesor" con riqueza en múltiples resoluciones temporales.

***

# Research Problem

**Problema de investigación:** Los modelos actuales de representación de audio musical sufren una disyuntiva: o emplean cuantización discreta eficiente en una escala temporal única (perdiendo dependencias estructurales a largo plazo), o modelan jerarquías musicales usando espacios latentes continuos (perdiendo la compresión, robustez e indexación eficiente de los enfoques discretos).
**Definición clara:** El desarrollo de una arquitectura de aprendizaje profundo que integre **cuantización vectorial (Codebooks)** con **modelado temporal multiescala** para crear una representación de audio musical compacta y discretizada en distintos niveles jerárquicos.
**Preguntas de investigación:**
1. ¿Cómo se puede estructurar un modelo de predicción de tokens enmascarados/contrastivo para aprender un libro de códigos (codebook) distribuido en diferentes resoluciones temporales?
2. ¿Mejora una representación jerárquica discreta el rendimiento simultáneo en tareas de comprensión musical a nivel de frame (ej. <i>pitch</i>, acordes) y a nivel estructural (ej. segmentación, género)?
**Hipótesis:** La integración de libros de códigos paralelos organizados jerárquicamente por escalas temporales producirá representaciones más robustas y generalizables, superando a los enfoques de codebook plano en tareas que dependen de la macro-estructura (como segmentación y recomendación), manteniendo la eficiencia computacional.

***

# Methodology

*   **Tipo de enfoque:** Deep Learning (DL), Aprendizaje Autosupervisado (Self-Supervised Learning - SSL). Combinación de modelado **Masked Token Prediction (MTP)** similar a BEST-RQ u OMAR-RQ y **Aprendizaje Contrastivo Multiescala**.
*   **Datasets (Existentes):** 
    *   *Preentrenamiento (No etiquetado):* The Free Music Archive (FMA) y/o audios recolectados sin etiqueta a gran escala.
    *   *Tareas Downstream (Evaluación):* MagnaTagATune (Auto-etiquetado), SALAMI y Harmonix (Segmentación estructural), OpenMIC (Clasificación de instrumentos), NSynth (Tono/Pitch).
*   **Pipeline del sistema:**
    1.  **Extracción de Entrada:** Transformación del audio crudo a características acústicas locales (ej. Mel-espectrogramas, CQT) extraídas en ventanas superpuestas (patches).
    2.  **Codificador Jerárquico (Encoder):** Una arquitectura basada en Transformers o Conformer que procesa la secuencia.
    3.  **Cuantización Multiescala (Multi-Scale Quantizer):** Uso de múltiples codebooks operando a diferentes tasas de muestreo latente (ej. nivel fino para 200ms, nivel grueso para 2s o a nivel de <i>beat</i>).
    4.  **Objetivo SSL:** Una pérdida conjunta que obliga a la red a predecir los tokens discretos enmascarados (MTP) a distintas escalas, combinado opcionalmente con pérdida contrastiva (N-pair loss) para agrupar frames cercanos en la misma sección funcional.
    5.  **Probing (Inferencia):** Congelación del modelo base y entrenamiento de un clasificador lineal (MLP superficial) sobre las representaciones aprendidas para cada tarea downstream.
*   **Métricas de evaluación:**
    *   *Clasificación/Etiquetado:* mean Average Precision (mAP), F1-score macro, ROC-AUC.
    *   *Segmentación:* HR.5F, HR3F (Hit-rate para detección de fronteras), V-measure, L-measure (para segmentación jerárquica multinivel).
*   **Estrategia de validación:** División estándar de los datasets (ej. partición 12:3:1 en MTAT), validación cruzada (K-fold) en tareas pequeñas, y pruebas de robustez contra corrupción de datos para validar la solidez de las características aprendidas. Se comparará estadísticamente el rendimiento contra <i>baselines</i> planos (flat) utilizando pruebas no paramétricas (ej. Wilcoxon signed-rank test).

***

# Expected Impact

**Científico:** Establecerá un nuevo paradigma al unificar la literatura de codificación neural discreta (Vector Quantization) y el análisis de estructura musical jerárquica. Sentará un precedente sobre cómo la información musical multiescala debe codificarse discretamente para modelos de base (Foundation Models) en audio.
**Aplicado (industria/sociedad):** Revolucionará sistemas industriales de gran escala (ej. plataformas de <i>streaming</i> musical). Las representaciones de libros de códigos discretos reducen drásticamente los costos de almacenamiento y permiten una búsqueda/indexación ultrarrápida. Mejorará directamente los sistemas de recomendación musical híbridos resolviendo problemas de "arranque en frío" (cold start) y permitiendo navegación estructural dentro de la canción.
**Ético:** Al ser un enfoque basado en software de código abierto (Open Source), democratizará el acceso a representaciones de audio potentes sin que los investigadores dependan de infraestructuras masivas de cómputo cerradas. Además, se prestará especial atención a la diversidad del dataset para evitar sesgos inherentes hacia géneros musicales occidentales.