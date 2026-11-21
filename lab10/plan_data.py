"""Genera la lista mínima de audio antes de cargar/obtener pesos. No calcula XAI."""

import json
from pathlib import Path

import pandas as pd

from lab10.audit import OUT, ROOT, TEST_SPLITS
from lab10.pipeline import sample_cases, settings_from_env
from src.config import load_config
from src.data.dataset import AudioDataset


def plan_data(settings):
    s = settings
    if s.model not in {"C2", "C3", "C4"}:
        raise ValueError("Definir LAB10_MODEL tras confirmar la versión final.")
    if not set(s.splits) <= set(TEST_SPLITS):
        raise ValueError("Solo se permiten splits de prueba.")
    runs = pd.read_csv(ROOT / "results/runs.csv").set_index("run")
    cfg = load_config(ROOT / runs.loc[s.model, "config"], s.machine)
    cfg["dataset_roots"].update(s.roots)
    thresholds = json.loads(Path(s.thresholds).read_text())
    pool = []
    for split in s.splits:
        ds = AudioDataset(cfg, split, train=False)
        proto = ds.protocol.copy()
        proto["dataset_index"] = range(len(proto))
        scores = pd.read_csv(ROOT / f"results/scores/{s.model}__{split}.csv")
        if set(proto.utt_id) != set(scores.utt_id):
            raise ValueError(f"IDs inconsistentes: {split}")
        joined = proto.merge(scores, on="utt_id", suffixes=("", "_scores"), validate="one_to_one")
        if not (joined.label == joined.label_scores).all():
            raise ValueError("Etiquetas inconsistentes.")
        joined["cached_score"] = joined.pop("score")
        joined["split"] = split
        joined["audio_path"] = [str(ds.roots[r.dataset] / r.path) for r in joined.itertuples()]
        pool.append(joined)
    OUT.mkdir(parents=True, exist_ok=True)
    local, global_sample = sample_cases(pd.concat(pool, ignore_index=True), s, thresholds)
    print(f"Plan {s.model}: 5 locales, {len(global_sample)} globales; ver lab10/outputs/required_audio.csv")
    print("Selección documental pendiente de verificar con audio y checkpoint; no son resultados nuevos.")
    return local, global_sample


if __name__ == "__main__":
    plan_data(settings_from_env())
