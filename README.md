# Detección de voz clonada en centros de contacto bancarios

Proyecto final de CC3092 Deep Learning y Sistemas Inteligentes (UVG).
Diego Rosales (23258) y Juan Francisco Martínez (23617).

El sistema analiza los primeros 4 segundos de voz de una llamada al centro de contacto de un
banco, estima la probabilidad de que la voz haya sido generada por IA y devuelve al agente una
de tres acciones: **continuar**, **pedir verificación adicional** o **transferir a un analista de
fraude**. Se comparan tres modelos (LFCC + LCNN, AASIST-L y XLS-R + AASIST) por su valor de
negocio en USD/mes, no solo por EER.

## Estructura

```
configs/            configs de cada modelo y de cada máquina (laptop, kaggle)
src/
  data/             datasets, protocolos y augmentations (códecs telefónicos, RawBoost)
  features/         LFCC
  models/           lfcc_cnn, aasist, xlsr_aasist
  value/            modelo de valor y análisis de sensibilidad
app/                API FastAPI y panel del agente
kaggle/             notebook que lanza el entrenamiento en Kaggle
scripts/            descarga de datos y utilidades
notebooks/          EDA y figuras
results/            métricas (runs.csv) y figuras
```

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y una GPU NVIDIA con driver reciente.

```bash
uv sync                      # Python 3.11 + PyTorch con CUDA 12.8 + dependencias
uv sync --group dev          # pytest, ruff, ipykernel
uv sync --group clonacion    # XTTS-v2, solo para generar los clones de voz propios
```

PyTorch se instala desde el índice `cu128`, que es el mínimo para GPUs Blackwell (RTX 50xx) y
también funciona en GPUs anteriores.

## Verificar la GPU

```bash
uv run python scripts/check_gpu_memory.py                              # laptop: batch 4, bf16
uv run python scripts/check_gpu_memory.py --batch-size 8 --precision fp16   # Kaggle (T4)
```

Corre unos pasos de fine-tuning de XLS-R 300M con audio aleatorio y reporta el pico de VRAM,
el tiempo por paso y una estimación del tiempo por época.

## Entrenamiento

```bash
uv run python -m src.train --config configs/aasist_l.yaml --machine laptop --name B1
uv run python -m src.train --resume runs/B1          # retomar si se cortó (p. ej. en Kaggle)
```

- Combina el config del modelo con `configs/machines/<máquina>.yaml` (precisión, batch, rutas).
- Mide el EER en dev al final de cada época, guarda `runs/<corrida>/last.pt` (para retomar) y
  `best.pt` (mejor EER), y se detiene si el EER deja de mejorar (`train.patience`).
- Al terminar agrega una fila a `results/runs.csv` con máquina, GPU, tiempo, costo y métricas.
- Curvas en TensorBoard: `uv run tensorboard --logdir runs`.

## Evaluación

```bash
uv run python -m src.evaluate --checkpoint runs/B1/best.pt --splits asv19_eval asv21_eval
```

Guarda los scores P(sintética) por audio en `results/scores/` (los usa el modelo de valor) y las
métricas en `results/metrics/`: EER y AUC con IC 95 % por bootstrap, min t-DCF en ASVspoof 2019
y EER por ataque, códec e idioma.

## Pruebas

```bash
uv run pytest
```

Prueban las métricas contra casos calculados a mano y el ciclo completo de entrenamiento
(CPU, bf16, fp16, retomar y evaluar) con un dataset sintético (`--fake-data`) y un modelo
mínimo (`configs/debug.yaml`).

## Datos

Ver [`data/README.md`](data/README.md). ASVspoof 2019 LA, ASVspoof 2021 LA, MLAAD (es) y
grabaciones propias con consentimiento.
