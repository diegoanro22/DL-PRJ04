# Datos

Esta carpeta no se versiona. En la laptop los datos se descargan aquí con
`scripts/download_data.sh`; en Kaggle se montan como datasets en `/kaggle/input/`.
La ruta de cada dataset se define en `configs/machines/*.yaml` (`dataset_roots`), y los
protocolos unificados están en `protocols/` (ver `src/data/README.md`).

Estructura esperada:

```
data/
  ASVspoof2019_LA/     # train, dev, eval + protocolos
  ASVspoof2021_LA/     # eval (canal telefónico y códecs)
  MLAAD_es/            # subconjunto en español + M-AILABS (es)
  propios/             # grabaciones del equipo y clones XTTS (con consentimiento)
```
