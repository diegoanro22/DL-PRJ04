# Kaggle

- Acelerador: **Nvidia Tesla T4** (internet on).
- Dataset: `awsaf49/asvpoof-2019-dataset`.
- Runner: `runner.ipynb` (parámetros `CONFIG` / `RUN_NAME` al inicio).
- Generar kernel: `python scripts/make_kaggle_kernel.py --run B1 --out /tmp/kaggle-b1`.

No hace falta subir el repo como dataset: el notebook clona `main` desde GitHub.
Los fixes de estabilidad (workers=0, skip-file-check, progreso) deben estar en `main`
antes de lanzar A2/B1/B2.
