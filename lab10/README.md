# Laboratorio 10 — Explainable AI · C4

[explainability.ipynb](explainability.ipynb) aplica oclusión temporal y por bandas a C4,
sin reentrenarlo. Genera las tablas y figuras del análisis en `outputs/`.

## Ejecución desde la raíz

Python 3.11 y `uv`:

```bash
uv sync --group dev
uv pip install --python .venv/bin/python -r lab10/requirements.txt
uv run --no-sync python -m lab10.acquire --onnx
uv run --no-sync python -m lab10.plan_data
uv run --no-sync python -m lab10.download_asv_sample
LAB10_THREADS=4 uv run --no-sync python -m lab10.run
```

`run` ejecuta el notebook completo en un kernel nuevo con el mismo Python. Al terminar,
guarda `outputs/explainability.executed.ipynb`, el manifiesto de salidas y el estado de
la ejecución. `--no-sync` conserva las dependencias adicionales del notebook.

Se requiere `ffmpeg` con `pcm_mulaw` y `libgsm` para las sondas de códec. CPU/float32 es
suficiente; CUDA es opcional. Se recomienda lote 1 por memoria. Los tiempos y versiones
medidos están en `outputs/environment.json` y las tablas de costo.

## Modelo y datos

C4 es la versión final publicada en
[r0sV/signet](https://huggingface.co/r0sV/signet/tree/fa67b890666c8f92477a27d4155032afbf86b405/C4).
`acquire --onnx` descarga únicamente sus archivos, fijados a la revisión
`fa67b890666c8f92477a27d4155032afbf86b405`, y valida los hashes:

- `best.pt`: `f83e759c2b4186bd9b7097a81c9c3fb2fde1a6745773e990ee2493ab8139afb0`.
- `model.onnx`: `a7e03a0410e9685e3cca6a3624910bc9ee9f15929bc1cc8d1a81d0b201c3750b`.

Ambos ocupan unos 2.53 GB; durante la descarga se necesita hasta 1.26 GB adicional.
La configuración es `configs/xlsr_aasist_frozen_es_omni.yaml`; la mejor época 14 figura
como `epoch=13` en el checkpoint. `assets/xlsr_config.json` permite cargar toda la
arquitectura sin descargar un segundo backbone.

El análisis usa **ASVspoof 2019 LA eval**, separado de entrenamiento: cinco casos locales
y 28 globales (2 genuinos y 2 por ataque A07–A19), semilla 42. La comparación entre clases
es exploratoria. C4 se entrenó también con español moderno, pero ese audio no está incluido
en esta evaluación.

El descargador obtiene solo los 33 FLAC de `outputs/required_audio.csv` del
[dataset público de Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset),
conservando las rutas del protocolo bajo `data/ASVspoof2019_LA/`. No requiere credenciales.
Los pesos y audios no se versionan. Deben respetarse sus licencias y el consentimiento
para cualquier grabación privada; C4 y sus fuentes sintéticas tienen restricciones de
uso comercial.

## Configuración

Las variables se definen antes de abrir el kernel:

```bash
export LAB10_MODEL=C4
export LAB10_CHECKPOINT=runs/C4/best.pt
export LAB10_THRESHOLDS=runs/C4/thresholds.json
export LAB10_SPLITS=asv19_eval
export LAB10_DEVICE=cpu
export LAB10_PRECISION=float32
export LAB10_BATCH=1
export LAB10_THREADS=4
export LAB10_PER_STRATUM=2
```

`LAB10_MACHINE` selecciona `configs/machines/<máquina>.yaml` (por defecto `laptop`).
Para otras rutas, `LAB10_DATASET_ROOTS` admite un JSON, por ejemplo
`'{"asv19":"/ruta/LA/LA","es":"/ruta/dataset-es"}'`. Al cambiar splits se debe generar
el plan y obtener los archivos originales indicados. No se sustituyen audios ausentes.

La entrada es mono a 16 kHz, con 64,600 muestras (4.0375 s), clamp [-1,1], recorte inicial
o repetición de cortos. Las acciones usan softmax crudo: continuar si p<0.957, verificar
en [0.957,0.9978) y analista desde 0.9978. Un negativo binario en t2 no equivale a continuar.
La calibración cambia la probabilidad mostrada, no esas acciones.

## Salidas

| Etapa | Archivos en `outputs/` |
|---|---|
| Configuración y auditoría | `environment.json`, `provenance.json`, `repository_audit.json`, `model_comparison.csv`, `split_integrity.csv`, `data_inventory.csv` |
| Selección y adquisición | `local_selection.csv`, `global_sample.csv`, `required_audio.csv`, `global_distribution.csv` |
| Verificación | `local_cases.csv`, `score_verification.csv`, `score_agreement.json`, `precision_audit.csv`, `precision_audit_timing.json`, `prediction_timings.csv` |
| Contraste de despliegue | `deployment_verification.csv`, `runtime_action_changes.csv`, `deployment_summary.json` |
| Atribuciones y controles | `temporal.csv`, `local_bands.csv`, `local_timings.csv`, `perturbation_qc.csv`, `fidelity.csv`, `risk_probes.csv` |
| Agregados e interpretación | `global_attributions.csv`, `global_summary.csv`, `global_timing.json`, `local_cases_table.csv`, `probe_action_changes.csv`, `probe_directions.csv`, `sampling_summary.json`, `fidelity_control_summary.csv` |
| Diseño y costo | `methods.csv`, `parameters.csv`, `findings.json`, `stage_costs.csv` |
| Figuras | `local_explanations.png`, `global_explanations.png`, `fidelity.png` |
| Ejecución | `explainability.executed.ipynb`, `artifact_manifest.json`, `execution_status.json` |

`missing_audio.csv` aparece solo si faltan archivos. El manifiesto registra hashes de
las salidas; el estado distingue ejecución completa de una falla.

## Reproducibilidad y errores comunes

La semilla 42 fija `random`, NumPy y PyTorch, el muestreo y los controles. Se desactiva
TF32 y se solicitan algoritmos deterministas con advertencias. Hardware, hilos y versiones
pueden producir pequeñas diferencias numéricas; los tiempos también pueden variar.

Los CSV GPU/bf16 históricos no se reproducen exactamente en CPU/float32. Se mantienen
los desacuerdos con atol=0.005 y las acciones de ambos entornos. El ONNX publicado se
comprueba usando `app.inference.ServedModel`, con los dos hilos predeterminados de la API,
sobre las mismas 33 ventanas originales; no se consulta un servicio remoto.

- **Pesos o datos ausentes:** ejecutar adquisición/plan y revisar las rutas del manifiesto.
- **Hash o época distintos:** revisar el checkpoint; no cambiar tolerancias ni umbrales.
- **Memoria insuficiente:** usar CPU y lote 1 sin reducir casos ni duración.
- **Códec no disponible:** instalar `ffmpeg` con los codificadores indicados.
- **Ejecución interrumpida:** consultar `execution_status.json`; las salidas antiguas no
  acreditan la ejecución nueva.

En modo interactivo puede utilizarse Jupyter con el kernel del entorno `.venv`: reiniciar
el kernel y ejecutar todas las celdas en orden, después de definir las variables.
