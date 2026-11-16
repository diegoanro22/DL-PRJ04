# Modelos servidos

Cada subcarpeta `<nombre>/` tiene lo que la API necesita para servir un modelo:
`model.onnx`, `calibration.json` y `thresholds.json`. No se versionan en git (el ONNX de C2
pesa ~1.3 GB); se generan con:

```bash
uv run python -m app.package_model --checkpoint runs/C2/best.pt --name C2 \
    --t1 0.9258 --t2 0.9258 --tasa-alertas 0.00277
```

Para construir la imagen sin el checkpoint, el `Dockerfile` puede bajarlos de un repo de
modelos de Hugging Face con la misma estructura (`C2/model.onnx`, ...):

```bash
docker build --build-arg MODEL_REPO=<usuario>/<repo> -t signet .
```
