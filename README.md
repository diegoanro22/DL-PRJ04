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
| A1 | `configs/lfcc_cnn.yaml` | LFCC + LCNN desde cero |
| A2 | `configs/lfcc_cnn_aug.yaml` | A1 + telephony + RawBoost |
| B1 | `configs/aasist_l.yaml` | AASIST-L (85k parámetros) desde cero |
| B2 | `configs/aasist_l_aug.yaml` | B1 + telephony + RawBoost |
| C0 | `configs/xlsr_aasist_frozen.yaml` | XLS-R 300M congelado + cabeza AASIST (modelo base del transfer learning) |
| C1 | `configs/xlsr_aasist.yaml` | Fine-tuning completo de XLS-R 300M + cabeza AASIST |
| C1b | `configs/xlsr_aasist_lpft.yaml` | LP-FT: primero solo la cabeza, luego todo |
| C2 | `configs/xlsr_aasist_frozen_aug.yaml` | C0 + telephony y RawBoost (cada una con probabilidad 0.5); modelo servido |

La iteración 2 de cada modelo (A2, B2, C2) activa `augment.telephony` y `augment.rawboost`
(`true` en todos los audios o una probabilidad entre 0 y 1 por audio). C0 sirve para medir
cuánto aporta el fine-tuning: C1 y C1b quedaron peor que C0 en eval, así que C2 parte de C0.

### Modelo de valor

```bash
uv run python -m src.value.value_model \
  --scores results/scores/C0__asv19_dev.csv results/scores/C1__asv19_dev.csv \
  --names C0 C1 --out results/value
uv run python -m src.value.sensitivity --scores results/scores/C0__asv19_dev.csv --name C0
```

Elige umbrales t1 < t2 por valor en USD/mes (no por EER), guarda curvas y un resumen CSV.
La sensibilidad produce el tornado ±20/50 % y el heatmap prevalencia × L.

### Kaggle

El notebook `kaggle/runner.ipynb` clona el repo, instala deps, genera protocolos y lanza
`python -m src.train --config ... --machine kaggle`. Preferir acelerador **GPU T4**.

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

## API

```bash
uv run python -m app.package_model --checkpoint runs/C2/best.pt --name C2 --t1 0.42 --t2 0.97
uv run uvicorn app.api:app --reload
curl -F audio=@llamada.wav "localhost:8000/analyze?modelo=C2"
```

| Ruta | Qué hace |
|---|---|
| `POST /analyze` | Recibe audio (wav, flac, webm, ogg, mp3), usa los primeros 4 s y devuelve el score, P(sintética) calibrada para la prevalencia indicada, la acción (continuar, verificar o analista), avisos de calidad (audio corto, silencio, saturación) y la latencia |
| `GET /models` | Modelos disponibles en `app/models/` con sus umbrales |
| `GET /health` | Estado del servicio |
| `GET /metrics` | Llamadas y acciones por modelo, latencia p50/p95, histograma de scores y alarma de deriva cuando la tasa de alertas se aleja de la esperada |

`app/package_model.py` exporta el checkpoint a ONNX y copia la calibración y los umbrales a
`app/models/<nombre>/`. La API corre solo con ONNX Runtime en CPU (sin PyTorch) y no guarda
audio. Sin `--t1/--t2` marca los umbrales como provisionales. Webm, ogg y mp3 requieren ffmpeg.

### Panel del agente

`app/web/` es el panel (React + Vite) que usa el agente: graba del micrófono o recibe un
archivo, dibuja la forma del audio con la ventana de 4 s que ve el modelo, y muestra la acción,
la probabilidad calibrada, dónde cayó el score entre los umbrales, los avisos de calidad y un
registro de la sesión. El build va a `app/static/` y la API lo sirve en `/`.

```bash
cd app/web && pnpm install && pnpm build   # genera app/static
uv run uvicorn app.api:app                  # panel en http://localhost:8000
# Desarrollo con recarga: API en :8000 y `pnpm dev` en app/web (Vite hace de proxy)
```

### Docker y Hugging Face Spaces

El `Dockerfile` construye el panel y arma una imagen de inferencia sin PyTorch (FastAPI +
ONNX Runtime + ffmpeg) con los modelos de `app/models/`, en el puerto 7860.

```bash
uv run python -m app.package_model --checkpoint runs/C2/best.pt --name C2 --t1 0.9258 --t2 0.9258 --tasa-alertas 0.00277
docker build -t verificador-voz .
docker run -p 7860:7860 verificador-voz     # http://localhost:7860
```

Los modelos no están en git (el ONNX de C2 pesa ~1.3 GB): la imagen usa `app/models/C2` si se
empaquetó localmente o, con `--build-arg MODEL_REPO=<usuario>/<repo>`, lo baja de un repo de
modelos de Hugging Face con la misma estructura. Sin ninguno de los dos, el build falla con un
mensaje claro. La imagen solo incluye C2 (ver `.dockerignore`). Para HF Spaces se crea un Space
con SDK Docker; su README necesita `sdk: docker` y `app_port: 7860`.

## Pruebas

```bash
uv run pytest
```

Prueban las métricas y la calibración contra casos calculados a mano, el ciclo completo de
entrenamiento (CPU, bf16, fp16, retomar y evaluar) con un dataset sintético (`--fake-data`),
los modelos AASIST-L y XLS-R + AASIST (congelado y con fine-tuning, con un wav2vec2 chico para
no bajar XLS-R), la exportación a ONNX y la API (formatos de audio, avisos, umbrales,
errores y alarma de deriva).

## Datos

Ver [`data/README.md`](data/README.md). ASVspoof 2019 LA, ASVspoof 2021 LA, MLAAD (es) y
grabaciones propias con consentimiento.

Antes de entrenar con ASVspoof 2019 LA, genera los protocolos unificados usando la misma
carpeta que figura en `dataset_roots.asv19` del config de tu máquina:

```bash
uv run python scripts/build_protocols.py --root data/ASVspoof2019_LA
# En Kaggle (omitir el check por archivo: el FS de input cuelga con miles de exists()):
uv run python scripts/build_protocols.py --root /kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA --skip-file-check
```

El script genera `protocols/asv19_{train,dev,eval}.csv` con rutas relativas, valida etiquetas,
IDs únicos y existencia de los FLAC. Acepta las carpetas oficiales `ASVspoof2019_LA_train/`
(y dev/eval) o los nombres cortos `train/`, `dev/`, `eval/`. Los archivos oficiales
`ASVspoof2019.LA.cm.*.txt` deben estar en `ASVspoof2019_LA_cm_protocols/` o en la raíz del
dataset. Usa `--splits train dev` si todavía no descargaste eval. Generar su CSV solo lee
metadatos; no ejecuta la evaluación del modelo.

### Kaggle (T4)

Notebook canónico: [`kaggle/runner.ipynb`](kaggle/runner.ipynb). Para generar un kernel
A1/A2/B1/B2 listo para push:

```bash
python scripts/make_kaggle_kernel.py --run B1 --out /tmp/kaggle-b1
kaggle kernels push -p /tmp/kaggle-b1 --accelerator NvidiaTeslaT4
```

Gotchas ya cubiertos en el runner / `configs/machines/kaggle.yaml`:

| Problema | Mitigación |
|---|---|
| Path del dataset en `/kaggle/input/datasets/owner/slug/...` | `find_asv19()` (sin `rglob` profundo) |
| `build_protocols` cuelga comprobando cada FLAC | `--skip-file-check` |
| `LibsndfileError` / eval opaca con workers | `num_workers: 0` (+ fallback torchaudio en el dataloader) |
| Loss NaN en batch 1 con fp16 | LFCC en fp32 (`src/features/lfcc.py`) |
| Log parece “stuck” horas | prints mid-epoch / mid-eval con `flush=True` |
| P100: “no kernel image” | Acelerador **T4**, no P100 |

El dataloader convierte a mono y 16 kHz, repite audios cortos y entrega 64,600 muestras:
recorte aleatorio en train y el inicio en dev/test, conservando la amplitud sin normalizar.
Las augmentations `telephony` (8 kHz + pasa-banda 300–3400 Hz) y `rawboost` se activan con
los configs `*_aug.yaml`. Los códecs G.711/GSM/AMR se precalculan con
`scripts/precompute_codecs.py` (no al vuelo). Los generadores de protocolos de ASVspoof 2021,
español y datos propios corresponden a etapas posteriores.
