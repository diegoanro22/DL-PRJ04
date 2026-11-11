"""Prepara un modelo entrenado para servirlo en la API: app/models/<nombre>/.

Exporta el checkpoint a ONNX (verificando que coincida con PyTorch), copia la calibración si
existe y escribe los umbrales. Sin --t1/--t2 la API los marca como provisionales.

Uso:
    uv run python -m app.package_model --checkpoint runs/C2/best.pt --name C2 \\
        --t1 0.42 --t2 0.97 --tasa-alertas 0.012
"""

import argparse
import json
import shutil
from pathlib import Path

from src.export_onnx import export, load_model

MODELS_DIR = Path(__file__).resolve().parent / "models"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="runs/<corrida>/best.pt")
    p.add_argument("--name", help="nombre del modelo en la API (por defecto, la corrida)")
    p.add_argument("--calibration", help="por defecto runs/<corrida>/calibration.json si existe")
    p.add_argument("--t1", type=float, help="umbral de verificación adicional (score crudo)")
    p.add_argument("--t2", type=float, help="umbral de transferencia al analista (score crudo)")
    p.add_argument("--tasa-alertas", type=float,
                   help="tasa de alertas esperada en producción, para la alarma de deriva")
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    name = args.name or ckpt.parent.name
    out = Path(args.models_dir) / name
    out.mkdir(parents=True, exist_ok=True)

    model, _, _ = load_model(ckpt)
    diff = export(model, out / "model.onnx")
    print(f"ONNX: {out / 'model.onnx'} (diferencia con PyTorch {diff:.1e})")

    calibration = Path(args.calibration) if args.calibration else ckpt.parent / "calibration.json"
    if calibration.exists():
        shutil.copy(calibration, out / "calibration.json")
        print(f"Calibración: {calibration}")
    else:
        # Si había una calibración de un empaquetado anterior, ya no corresponde a este modelo.
        (out / "calibration.json").unlink(missing_ok=True)
        print("Sin calibración: la API devolverá solo el score crudo.")

    if (args.t1 is None) != (args.t2 is None):
        raise SystemExit("Hay que pasar --t1 y --t2 juntos.")
    if args.t1 is not None:
        if args.t1 > args.t2:
            raise SystemExit("Se requiere t1 <= t2.")
        thresholds = {"t1": args.t1, "t2": args.t2, "tasa_alertas_esperada": args.tasa_alertas}
        (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
        print(f"Umbrales: {thresholds}")
    else:
        (out / "thresholds.json").unlink(missing_ok=True)
        print("Sin umbrales: la API usará t1 = 0.5 y t2 = 0.9 como provisionales.")


if __name__ == "__main__":
    main()
