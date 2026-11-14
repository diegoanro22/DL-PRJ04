#!/usr/bin/env bash
# Descarga / organiza los datasets en data/ para la laptop.
# ASVspoof 2019 LA suele tomarse del dataset público de Kaggle; 2021, MLAAD (es)
# y propios se suben como datasets privados del equipo.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"

echo "==> Destino: $DATA"
echo "ASVspoof 2019 LA (Kaggle público 'asvpoof-2019-dataset'):"
echo "  kaggle datasets download -d awsaf49/asvpoof-2019-dataset -p $DATA --unzip"
echo "  Luego: ln -sfn \$PWD/LA/LA $DATA/ASVspoof2019_LA   # o mueve la carpeta LA"
echo
echo "ASVspoof 2021 LA eval (dataset privado del equipo):"
echo "  kaggle datasets download -d <equipo>/asvspoof2021-la -p $DATA --unzip"
echo
echo "MLAAD (es) + M-AILABS (es):"
echo "  kaggle datasets download -d <equipo>/mlaad-es -p $DATA --unzip"
echo
echo "Grabaciones propias + clones XTTS:"
echo "  kaggle datasets download -d <equipo>/voces-propias -p $DATA --unzip"
echo "  o copia local a $DATA/propios/"
echo
echo "Cuando ASVspoof 2019 esté en $DATA/ASVspoof2019_LA:"
echo "  uv run python scripts/build_protocols.py --root data/ASVspoof2019_LA"
echo
echo "Nota: este script no descarga nada solo; imprime los comandos porque"
echo "varios datasets son privados y requieren 'kaggle' autenticado."
