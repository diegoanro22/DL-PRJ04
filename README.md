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

### Modelos

| Iteración | Config | Qué entrena |
|---|---|---|
| A1, A2 | `configs/lfcc_cnn.yaml` | LFCC + LCNN desde cero |
| B1, B2 | `configs/aasist_l.yaml` | AASIST-L (85k parámetros) desde cero |
| C0 | `configs/xlsr_aasist_frozen.yaml` | XLS-R 300M congelado + cabeza AASIST (modelo base del transfer learning) |
| C1, C2 | `configs/xlsr_aasist.yaml` | Fine-tuning completo de XLS-R 300M + cabeza AASIST |

La iteración 2 de cada modelo (A2, B2, C2) activa `augment.telephony` y `augment.rawboost`.
C0 sirve para medir cuánto aporta el fine-tuning: C0 → C1 → C2 se comparan en EER y valor.

## Evaluación

```bash
uv run python -m src.evaluate --checkpoint runs/B1/best.pt --splits asv19_eval asv21_eval
```

Guarda los scores P(sintética) por audio en `results/scores/` (los usa el modelo de valor) y las
métricas en `results/metrics/`: EER y AUC con IC 95 % por bootstrap, min t-DCF en ASVspoof 2019
y EER por ataque, códec e idioma.

## Calibración, ONNX y latencia

```bash
uv run python -m src.calibrate --scores results/scores/C1__asv19_dev.csv   # -> runs/C1/calibration.json
uv run python -m src.export_onnx --source runs/C1/best.pt                   # -> runs/C1/model.onnx
uv run python scripts/benchmark_latency.py --models runs/B2/best.pt runs/C2/best.pt
```

- La calibración de Platt convierte el score en P(sintética) para cualquier prevalencia, sin
  cambiar EER, TPR ni FPR.
- El benchmark mide latencia p50/p95 en ONNX Runtime con 2 hilos de CPU (la instancia donde se
  serviría), en PyTorch CPU y GPU, y el costo por 1,000 llamadas que usa el modelo de valor
  (`c_comp`). Guarda `results/latency.csv`. Acepta configs `.yaml` para medir antes de entrenar.

## Pruebas

```bash
uv run pytest
```

Prueban las métricas y la calibración contra casos calculados a mano, el ciclo completo de
entrenamiento (CPU, bf16, fp16, retomar y evaluar) con un dataset sintético (`--fake-data`),
los modelos AASIST-L y XLS-R + AASIST (congelado y con fine-tuning, con un wav2vec2 chico para
no bajar XLS-R) y la exportación a ONNX.

## Datos

Ver [`data/README.md`](data/README.md). ASVspoof 2019 LA, ASVspoof 2021 LA, MLAAD (es) y
grabaciones propias con consentimiento.

Antes de entrenar con ASVspoof 2019 LA, genera los protocolos unificados usando la misma
carpeta que figura en `dataset_roots.asv19` del config de tu máquina:

```bash
uv run python scripts/build_protocols.py --root data/ASVspoof2019_LA
# En Kaggle:
uv run python scripts/build_protocols.py --root /kaggle/input/asvpoof-2019-dataset/LA/LA
```

El script genera `protocols/asv19_{train,dev,eval}.csv` con rutas relativas, valida etiquetas,
IDs únicos y existencia de los FLAC. Acepta las carpetas oficiales `ASVspoof2019_LA_train/`
(y dev/eval) o los nombres cortos `train/`, `dev/`, `eval/`. Los archivos oficiales
`ASVspoof2019.LA.cm.*.txt` deben estar en `ASVspoof2019_LA_cm_protocols/` o en la raíz del
dataset. Usa `--splits train dev` si todavía no descargaste eval. Generar su CSV solo lee
metadatos; no ejecuta la evaluación del modelo.

El dataloader convierte a mono y 16 kHz, repite audios cortos y entrega 64,600 muestras:
recorte aleatorio en train y el inicio en dev/test, conservando la amplitud sin normalizar.
Las augmentations telefónicas y RawBoost todavía están pendientes; para esta etapa deja
ambos flags en `false`. Los generadores de protocolos de ASVspoof 2021, español y datos propios
corresponden a etapas posteriores.
