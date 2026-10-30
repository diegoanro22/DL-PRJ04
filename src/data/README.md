# Contrato de datos

Acuerdo entre el código de datos (Cisco) y `train.py` / `evaluate.py` (Diego). Si algo de aquí
cambia, se avisa al otro antes de hacer merge.

## 1. Protocolos unificados

Cada split tiene un CSV en `protocols/` (raíz del repo, **versionado en git**), generado por
`scripts/build_protocols.py` a partir de los protocolos oficiales de cada dataset.

```
protocols/asv19_train.csv  asv19_dev.csv  asv19_eval.csv
          asv21_eval.csv   mlaad_es.csv   propios.csv
```

| Columna | Ejemplo | Notas |
|---|---|---|
| `utt_id` | `LA_T_1138215` | Único dentro del CSV |
| `dataset` | `asv19` | Clave en `dataset_roots` del config de la máquina |
| `path` | `train/flac/LA_T_1138215.flac` | Relativo a `dataset_roots[dataset]` |
| `label` | `1` | **1 = sintética, 0 = genuina** |
| `attack` | `A07` | `-` si es genuina |
| `speaker` | `LA_0079` | `-` si no se conoce |
| `language` | `en` | ISO 639-1 |
| `codec` | `none` | `none`, `g711`, `gsm`, `amr` o el que traiga ASVspoof 2021 |

El nombre del split es el nombre del archivo sin `.csv` (`"asv19_train"`).

## 2. Rutas por máquina

`configs/machines/*.yaml` define `dataset_roots`, la carpeta raíz de cada dataset:

```yaml
dataset_roots:
  asv19: data/ASVspoof2019_LA            # laptop
  asv19: /kaggle/input/<dataset>/LA      # kaggle
```

## 3. Qué devuelve una muestra

```python
{"wav": torch.Tensor,   # float32, shape [64600] (4 s a 16 kHz)
 "label": int,          # 1 = sintética, 0 = genuina
 "utt_id": str}
```

Los metadatos (`attack`, `language`, `codec`...) no pasan por el dataloader: `evaluate.py` los
cruza con el CSV por `utt_id`.

## 4. Preparación del audio

- Mono, 16 kHz (remuestrear si viene en otra frecuencia), valores en [-1, 1] **sin normalizar**.
- **Train:** recorte aleatorio de 64,600 muestras; si es más corto, se repite hasta llenarlo.
- **Dev y test:** las **primeras** 64,600 muestras (repetir si es más corto). Imita el caso real:
  se decide con el inicio de la llamada.
- Todos los modelos reciben la forma de onda. Los LFCC del modelo A se calculan **dentro del
  modelo** (primera capa), no en el dataloader.

## 5. Augmentation (solo en train)

Firma común, en `src/data/augment.py`:

```python
def fn(wav: np.ndarray, sr: int) -> np.ndarray: ...
```

Se activan con los flags del config del modelo:

| Flag | Qué aplica | Cuándo |
|---|---|---|
| `augment.telephony` | 8 kHz y de vuelta a 16 kHz, pasa-banda 300 a 3400 Hz, y con cierta probabilidad un audio precalculado con códec | Al vuelo; los códecs vienen precalculados |
| `augment.rawboost` | RawBoost (Tak et al., 2022) | Al vuelo |

Los códecs (G.711, GSM, AMR-NB) **no** se aplican al vuelo, porque ffmpeg por muestra es muy
lento. Se precalculan una vez con `scripts/precompute_codecs.py` sobre un subconjunto de train y
quedan como el split `asv19_train_codec`.

## 6. Convención del score

Todo el proyecto usa **score = P(sintética)**: más alto significa más probable que sea falsa.
Los scripts oficiales de ASVspoof usan la convención contraria (más alto = genuina);
`evaluate.py` invierte el score solo al llamarlos.

## 7. Funciones que usa `train.py`

Están definidas en `src/data/dataset.py`, con la firma ya fijada:

```python
build_dataloader(cfg: dict, split: str, train: bool) -> DataLoader
class_weights(split: str) -> torch.Tensor   # [peso_genuina, peso_sintética]
```

`cfg` es el config del modelo ya combinado con el de la máquina. Las claves que se usan son
`dataset_roots`, `num_workers`, `audio.sample_rate`, `audio.num_samples`, `train.batch_size`,
`augment.telephony` y `augment.rawboost`.
