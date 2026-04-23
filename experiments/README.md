# 🎵 SemiSupCon: Semi-Supervised Contrastive Learning of Musical Representations

> Implementación MVP del paper de **Julien Guinot, Elio Quinton y György Fazekas** (ISMIR 2024).

---

## 📄 ¿De qué trata el paper?

### El problema

El aprendizaje contrastivo auto-supervisado (self-supervised) ha demostrado gran potencial en Music Information Retrieval (MIR), pero tiene una limitación fundamental: al depender únicamente de cadenas de augmentación para definir qué muestras son "similares", el modelo puede aprender representaciones que **no capturan información musical relevante** para tareas downstream como clasificación de género, tagging o estimación de pitch.

### La propuesta: SemiSupCon

SemiSupCon propone una solución elegante y arquitecturalmente simple: **combinar objetivos contrastivos supervisados y auto-supervisados** en un marco semi-supervisado. La idea clave es:

- **Muestras no etiquetadas ($\mathcal{U}$)**: los "positivos" son únicamente las versiones augmentadas de la misma muestra (como en SimCLR).
- **Muestras etiquetadas ($\mathcal{S}$)**: los "positivos" incluyen **todas las muestras de la misma clase** en el batch, además de las augmentaciones.

Esto permite inyectar conocimiento musical (géneros, tags, instrumentación) directamente en el espacio de representaciones aprendido, sin necesidad de rediseñar la arquitectura.

### La pérdida SemiSupCon (Ecuación 3)

$$\mathcal{L}_{sem} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{k \neq i} \exp(z_i \cdot z_k / \tau)}$$

Donde $P(i)$ se define de forma adaptativa:

| Tipo de muestra | Positivos $P(i)$ |
|---|---|
| $i \in \mathcal{S}$ (supervisada) | Misma clase + augmentation pair |
| $i \in \mathcal{U}$ (no supervisada) | Solo augmentation pair |

### Contribuciones del paper

1. **Extensión simple** del contrastive learning al caso semi-supervisado, compatible con múltiples tipos de supervisión (labels binarios, multiclase, multilabel, regresión).
2. **Moldeo de representaciones** según la señal de supervisión elegida, con mínima pérdida de rendimiento en otras tareas.
3. **Robustez en regímenes de pocos datos** y mayor resistencia a corrupción de audio.

---

## 🔧 ¿Cómo se implementó?

### Arquitectura del pipeline

```
Audio MP3 (FMA Small)
    │
    ▼
Waveform Mono @ 22,050 Hz
    │
    ▼
Random Crop 2.7s (59,535 samples)
    │
    ├──► Augmentation View 1 ──┐
    │                          │
    └──► Augmentation View 2 ──┤
                               ▼
                    SampleCNN Encoder (d_E = 512)
                               │
                               ▼
                    Projection Head (d_g = 128)
                               │
                               ▼
                      L2 Normalization
                               │
                               ▼
                    SemiSupCon Loss (τ = 0.1)
                          ▲         ▲
                          │         │
                   Labels (S)   Sin labels (U)
```

### Componentes implementados

#### 1. Datos: FMA Small

- **Dataset**: [Free Music Archive (FMA)](https://github.com/mdeff/fma) — subset "small" (8,000 pistas, 8 géneros balanceados).
- **Muestreo**: 500 pistas seleccionadas aleatoriamente.
- **Split semi-supervisado**: 250 pistas con etiqueta de género ($\mathcal{S}$) + 250 sin etiqueta ($\mathcal{U}$).
- **Etiquetas**: Se usan los géneros top-level de FMA como proxy de tags de Last.fm (funcionalmente equivalente al paper).

| Género | Descripción |
|---|---|
| Electronic | Música electrónica |
| Experimental | Experimental/avant-garde |
| Folk | Folk/acústico |
| Hip-Hop | Hip-Hop/Rap |
| Instrumental | Instrumental |
| International | Música del mundo |
| Pop | Pop/mainstream |
| Rock | Rock/alternativo |

#### 2. Encoder: SampleCNN

Arquitectura end-to-end que opera sobre el waveform crudo (sin espectrogramas):

| Capa | Tipo | Channels | Kernel | Pool |
|---|---|---|---|---|
| 0 | Conv1D (stride=3) | 1 → 128 | 3 | — |
| 1–3 | Conv1D + BN + ReLU + MaxPool | 128 → 128 | 3 | 3 |
| 4–6 | Conv1D + BN + ReLU + MaxPool | 128/256 → 256 | 3 | 3 |
| 7–9 | Conv1D + BN + ReLU + MaxPool | 256 → 256/512 | 3 | 3 |
| GAP | AdaptiveAvgPool1d | 512 → 512 | — | — |

- **Entrada**: `(batch, 59535)` — waveform mono de 2.7 segundos
- **Salida**: `(batch, 512)` — embedding
- **Parámetros**: ~1.4M

#### 3. Projection Head

MLP de 2 capas con normalización L2:

```
Linear(512, 512) → ReLU → Linear(512, 128) → L2-Norm
```

#### 4. Augmentation Pipeline (Table 1 del paper)

| Augmentation | Parámetros | p |
|---|---|---|
| **Gain** | -15 a +5 dB | 0.5 |
| **Polarity Inversion** | Flip de polaridad | 0.5 |
| **Colored Noise** | White/Pink/Brown, SNR 3-30 dB | 0.5 |
| **Low-pass Filter** | Butterworth 2do orden, cutoff aleatorio | 0.5 |
| **High-pass Filter** | Butterworth 2do orden, cutoff aleatorio | 0.5 |
| **Band-pass Filter** | Center + bandwidth aleatorios | 0.5 |
| **Band-cut Filter** | Center + bandwidth aleatorios | 0.5 |
| **Pitch Shifting** | -4 a +4 semitonos (librosa) | 0.5 |

Cada muestra genera 2 vistas augmentadas independientes para el contrastive learning.

#### 5. Entrenamiento

| Hiperparámetro | Valor |
|---|---|
| Epochs | 50 |
| Batch size | 32 |
| Optimizador | Adam |
| Learning rate | $1 \times 10^{-4}$ |
| Temperatura $\tau$ | 0.1 |
| Proporción supervisada $b_s$ | 0.5 |
| Device | MPS (Apple Silicon) |

#### 6. Evaluación: Linear Probing

Para evaluar la calidad de las representaciones aprendidas:

1. **Congelar** el encoder SampleCNN
2. **Extraer** embeddings (512-dim) de todas las pistas
3. **Entrenar** un MLP clasificador:
   ```
   Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 8)
   ```
4. **Métricas**: Accuracy, F1-score (macro/weighted), confusion matrix por género

---

## 🚀 Cómo ejecutar

### Requisitos

- Python 3.10+
- macOS con Apple Silicon (MPS) o CUDA GPU
- ~8 GB de espacio en disco para datos

### Setup

```bash
# El venv ya fue creado
cd experiments

# Activar entorno virtual
source venv/bin/activate

# Abrir el notebook
jupyter notebook semisupcon_mvp.ipynb
```

### En VS Code

1. Abrir `semisupcon_mvp.ipynb`
2. Seleccionar kernel **"Python 3 (SemiSupCon)"**
3. Ejecutar todas las celdas secuencialmente

> **Nota**: La primera ejecución descarga ~7.5 GB de datos (FMA metadata + audio). Las ejecuciones posteriores reutilizan los datos descargados.

### Tiempo estimado

| Fase | Duración (MPS) | Duración (CPU) |
|---|---|---|
| Descarga de datos | ~10-30 min | ~10-30 min |
| Entrenamiento (50 epochs) | ~30-60 min | ~3-4 hrs |
| Linear Probing | ~2 min | ~5 min |

---

## 📂 Estructura del proyecto

```
experiments/
├── README.md                  # Este archivo
├── semisupcon_mvp.ipynb       # Notebook principal
├── venv/                      # Entorno virtual Python
└── data/                      # (generado en ejecución)
    ├── fma_metadata.zip
    ├── fma_small.zip
    ├── fma_metadata/
    │   ├── tracks.csv
    │   ├── genres.csv
    │   └── ...
    └── fma_small/
        ├── 000/
        ├── 001/
        └── ...
```

Archivos generados tras el entrenamiento:
```
experiments/
├── best_semisupcon.pt         # Checkpoint del mejor modelo
├── training_loss.png          # Gráfica de loss
└── probing_results.png        # Resultados del probing
```

---

## 📚 Referencias

- **Paper original**: Guinot, J., Quinton, E., & Fazekas, G. (2024). *Semi-Supervised Contrastive Learning of Musical Representations*. ISMIR 2024. [arXiv:2407.13840](https://arxiv.org/abs/2407.13840)
- **Código oficial**: [github.com/Pliploop/SemiSupCon](https://github.com/Pliploop/SemiSupCon)
- **SampleCNN**: Lee, J., et al. (2017). *Sample-level Deep Convolutional Neural Networks for Music Auto-tagging Using Raw Waveforms*. [arXiv:1703.01789](https://arxiv.org/abs/1703.01789)
- **FMA Dataset**: Defferrard, M., et al. (2017). *FMA: A Dataset For Music Analysis*. ISMIR 2017. [arXiv:1612.01840](https://arxiv.org/abs/1612.01840)
- **SupCon Loss**: Khosla, P., et al. (2020). *Supervised Contrastive Learning*. NeurIPS 2020. [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
