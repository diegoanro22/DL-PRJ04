"""Etapas del notebook: audio real, checkpoint existente, sin entrenamiento."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lab10.audit import OUT, ROOT, TEST_SPLITS, sha256, write_json
from src.config import apply_machine
from src.data.dataset import AudioDataset, load_protocol
from src.explain.occlusion import BANDS_HZ, time_mask
from src.models import build_model

SR, LENGTH = 16000, 64600


@dataclass
class Settings:
    model: str
    decision: str
    checkpoint: str
    thresholds: str
    roots: dict
    splits: tuple[str, ...]
    device: str = "cpu"
    machine: str = "laptop"
    seed: int = 42
    batch: int = 1
    per_stratum: int = 2
    window_s: float = 0.25
    stride_s: float = 0.125
    score_atol: float = 0.005
    precision: str = "float32"
    threads: int = 2
    control_repeats: int = 5


def settings_from_env() -> Settings:
    model = os.environ.get("LAB10_MODEL", "C4")
    return Settings(
        model=model, decision=os.environ.get("LAB10_MODEL_REASON",
            "C4 por indicación del equipo y ficha de Hugging Face: modelo servido, "
            "adaptado a español moderno; mejora es_test a costa de ASVspoof."),
        checkpoint=os.environ.get("LAB10_CHECKPOINT", f"runs/{model}/best.pt"),
        thresholds=os.environ.get("LAB10_THRESHOLDS", f"runs/{model}/thresholds.json"),
        roots=json.loads(os.environ.get("LAB10_DATASET_ROOTS", "{}")),
        splits=tuple(os.environ.get("LAB10_SPLITS", "asv19_eval").split(",")),
        device=os.environ.get("LAB10_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        machine=os.environ.get("LAB10_MACHINE", "laptop"),
        batch=int(os.environ.get("LAB10_BATCH", "1")),
        per_stratum=int(os.environ.get("LAB10_PER_STRATUM", "2")),
        precision=os.environ.get("LAB10_PRECISION", "float32"),
        threads=int(os.environ.get("LAB10_THREADS", "2")),
    )


def seed_everything(s: Settings) -> dict:
    random.seed(s.seed)
    np.random.seed(s.seed)
    torch.manual_seed(s.seed)
    torch.set_num_threads(s.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    hardware = {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": next((line.split(":", 1)[1].strip() for line in
            Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),
            platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor(),
        "cpu_threads": s.threads,
        "device": s.device, "precision": s.precision,
        "gpu": torch.cuda.get_device_name() if s.device.startswith("cuda") else None,
        "versions": {p: importlib.metadata.version(p) for p in (
            "torch", "torchaudio", "numpy", "pandas", "transformers", "captum",
            "matplotlib", "nbclient", "nbformat")},
        "determinism": "warn_only; resultados pueden variar con hardware, kernels y versiones",
        "ffmpeg": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "environment.json", hardware)
    return hardware


def methods_table() -> pd.DataFrame:
    rows = [
        ["Oclusión temporal y por bandas", "Retirar segmento o banda y medir Δ de log-odds.",
         "Caja negra; reemplazo explícito; sin gradientes.",
         "1+32+6 entradas/local; lote B reduce llamadas, no entradas.",
         "Directa para XLS-R+AASIST; tiempo y banda audibles.",
         "Sensibilidad; fuera de distribución, ringing; no causal ni aditiva.",
         "Alta: qué tramo/banda cambia el score; elegida."],
        ["Integrated Gradients (IG)", "Integrar derivadas del score desde una línea base al audio.",
         "Gradientes extremo a extremo; línea base; m pasos.",
         "m forwards + m backwards (p. ej. 50, no medidos aquí).",
         "Viable al quitar no_grad de extract SIN cambiar pesos; no implementada aquí.",
         "Completitud relativa a baseline, con error numérico; baseline de silencio discutible.",
         "Media: detalle por muestra difícil; descartada por acceso/costo."],
        ["Grad-CAM en representación", "Ponderar mapas internos con gradientes de clase.",
         "Gradientes, activaciones y capa objetivo.",
         "Un forward/backward más hooks; costo no medido.",
         "LFCC+LCNN existente explica A; XLS-R exige adaptar capa/ejes de la cabeza.",
         "Mapa grueso, depende de capa; ReLU usual oculta evidencia negativa.",
         "Media: mapa sobre rasgos latentes; Grad-CAM de A no se traslada a C."],
    ]
    table = pd.DataFrame(rows, columns=["método", "idea", "requisitos", "costo",
                                        "compatibilidad", "límites", "audiencia_elección"])
    table.to_csv(OUT / "methods.csv", index=False)
    return table


def preflight(s: Settings) -> dict:
    """Falla temprano; jamás cae a datos aleatorios, YAML o mapas preexistentes."""
    issues = []
    if s.model not in {"C2", "C3", "C4"} or not s.decision.strip():
        issues.append("Definir LAB10_MODEL y LAB10_MODEL_REASON con evidencia de la versión defendida.")
    if Path(s.checkpoint).suffix != ".pt" or not Path(s.checkpoint).is_file():
        issues.append(f"Falta checkpoint entrenado .pt: {s.checkpoint}")
    if not Path(s.thresholds).is_file():
        issues.append(f"Faltan umbrales operativos propios de {s.model}: {s.thresholds}")
    if s.model == "C4" and not Path(s.checkpoint).with_name("model.onnx").is_file():
        issues.append("Falta C4/model.onnx para verificar despliegue: python -m lab10.acquire --onnx")
    if not s.splits or not set(s.splits) <= set(TEST_SPLITS):
        issues.append("LAB10_SPLITS solo permite asv19_eval, asv19_eval_codec, es_test.")
    if s.batch < 1 or s.per_stratum < 1:
        issues.append("batch y cantidad por estrato deben ser positivos.")
    if s.precision not in {"float32", "bfloat16"}:
        issues.append("LAB10_PRECISION debe ser float32 o bfloat16.")
    if s.precision == "bfloat16" and not s.device.startswith("cuda"):
        issues.append("Este flujo limita bfloat16 a CUDA; CPU usa float32.")
    record = {"status": "BLOCKED" if issues else "PREREQUISITES_OK",
              "issues": issues, "settings": asdict(s)}
    # Rutas del usuario se mantienen locales; este registro no contiene voces ni credenciales.
    write_json(OUT / "execution_status.json", record)
    if issues:
        raise FileNotFoundError("\n".join(issues))
    return record


def prepare(s: Settings):
    preflight(s)
    thresholds = json.loads(Path(s.thresholds).read_text())
    if thresholds.get("name", s.model) != s.model:
        raise ValueError("Los umbrales pertenecen a otro modelo; no reutilizar los de C2 en C4.")
    t1, t2 = (float(thresholds[k]) for k in ("t1", "t2"))
    if not 0 < t1 <= t2 < 1:
        raise ValueError("Se requieren umbrales sobre softmax crudo: 0 < t1 <= t2 < 1.")
    # El checkpoint contiene TODO XLS-R: se construye la misma arquitectura desde su
    # config oficial y se carga estrictamente cada tensor, sin descargar otros pesos.
    # mmap y assign evitan duplicar 1.26 GB en memoria. No hay fallback a pesos aleatorios.
    checkpoint = torch.load(s.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    cfg = checkpoint["cfg"]
    checkpoint_epoch = checkpoint.get("epoch")
    from copy import deepcopy
    construction = deepcopy(cfg)
    construction["model_args"]["ssl_config"] = json.loads(
        (ROOT / "lab10/assets/xlsr_config.json").read_text())
    with torch.device("meta"):
        model = build_model(construction)
    model.load_state_dict(checkpoint["model"], strict=True, assign=True)
    del checkpoint
    cfg = apply_machine(cfg, s.machine)
    cfg["dataset_roots"].update(s.roots)
    if cfg["model"] != "xlsr_aasist" or cfg.get("audio", {}).get("num_samples") != LENGTH:
        raise ValueError("Arquitectura o longitud distinta de la auditada; revisar antes de explicar.")
    if cfg["audio"]["sample_rate"] != SR:
        raise ValueError("La frecuencia de muestreo no coincide con 16 kHz.")
    if cfg.get("train_split", "") in s.splits:
        raise ValueError("Se intentó explicar el split de entrenamiento.")
    model.to(s.device).eval()
    datasets, pools, overlap_rows = {}, [], []
    train_ids, train_paths, train_speakers = set(), set(), set()
    train_split = pd.read_csv(ROOT / "results/runs.csv").set_index("run").loc[s.model, "train_split"]
    training = load_protocol(train_split)
    train_ids.update(training.utt_id)
    train_paths.update(zip(training.dataset, training.path))
    train_speakers.update(training.loc[~training.speaker.isin(["", "-"]), "speaker"])
    for split in s.splits:
        ds = AudioDataset(cfg, split, train=False)
        datasets[split] = ds
        proto = ds.protocol.copy()
        # Los códecs son derivados de IDs de eval, no observaciones independientes.
        source_ids = proto.utt_id.str.replace(r"_(g711|gsm|amr)$", "", regex=True)
        leaks = set(source_ids) & train_ids
        path_leaks = set(zip(proto.dataset, proto.path)) & train_paths
        if leaks or path_leaks:
            raise ValueError(f"Hay solapamiento train/test en {split}: {sorted(leaks)[:5]}")
        speakers = set(proto.loc[~proto.speaker.isin(["", "-"]), "speaker"])
        overlap_rows.append({"split": split, "train_split": train_split,
                             "id_overlap": len(leaks), "path_overlap": len(path_leaks),
                             "speaker_overlap": len(speakers & train_speakers),
                             "speaker_metadata_missing": int(proto.speaker.isin(["", "-"]).sum())})
        scores_path = ROOT / f"results/scores/{s.model}__{split}.csv"
        scores = pd.read_csv(scores_path)
        if scores.utt_id.duplicated().any() or set(scores.utt_id) != set(proto.utt_id):
            raise ValueError(f"Scores y protocolo no cubren exactamente los mismos IDs: {split}")
        proto["dataset_index"] = np.arange(len(proto))
        joined = proto.merge(scores, on="utt_id", suffixes=("", "_cached"), validate="one_to_one")
        if not (joined.label == joined.label_cached).all():
            raise ValueError(f"Etiquetas CSV/protocolo divergentes: {split}")
        joined["split"] = split
        joined["cached_score"] = joined.pop("score")
        joined["audio_path"] = [str(ds.roots[r.dataset] / r.path) for r in joined.itertuples()]
        # Mapeo solo por raíz+path de protocolo: nunca escoger el primer nombre parecido.
        joined["audio_exists"] = joined.audio_path.map(lambda p: Path(p).is_file())
        pools.append(joined)
        metrics = json.loads((ROOT / f"results/metrics/{s.model}__{split}.json").read_text())
        # La época y el cfg se contrastan; un hash publicado no existe en los CSV antiguos.
        if checkpoint_epoch is not None and metrics["epoch"] != checkpoint_epoch:
            raise ValueError(f"La época del checkpoint no corresponde a los scores: {split}")
    pool = pd.concat(pools, ignore_index=True)
    pd.DataFrame(overlap_rows).to_csv(OUT / "split_integrity.csv", index=False)
    inventory = pool.groupby(["split", "label", "attack", "codec"]).agg(
        n=("utt_id", "size"), available=("audio_exists", "sum")).reset_index()
    inventory.to_csv(OUT / "data_inventory.csv", index=False)
    checkpoint_hash = sha256(Path(s.checkpoint))
    # Para la corrida entregada, el hash LFS público identifica inequívocamente C4.
    if s.model == "C4":
        from lab10.acquire import CHECKSUMS
        if checkpoint_hash != CHECKSUMS["best.pt"]:
            raise ValueError("Los pesos C4 no coinciden con el checkpoint publicado indicado.")
    provenance = {
        "model": s.model, "model_decision": s.decision,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_config_path": cfg.get("config_path"),
        "checkpoint_model_args": cfg.get("model_args"),
        "checkpoint_audio": cfg.get("audio"),
        "model_hub": "https://huggingface.co/r0sV/signet",
        "model_hub_revision": "fa67b890666c8f92477a27d4155032afbf86b405",
        "backbone_config_revision": "1a640f32ac3e39899438a2931f9924c02f080a54",
        "spanish_audio_status": "No disponible: generado en el equipo del compañero; "
            "no se regeneró ni se sustituyó por otro audio. XAI solo sobre splits indicados.",
        "thresholds": {"t1": t1, "t2": t2}, "thresholds_sha256": sha256(Path(s.thresholds)),
        "train_split": train_split,
        "split_hashes": {sp: sha256(ROOT / f"protocols/{sp}.csv") for sp in s.splits},
        "score_hashes": {sp: sha256(ROOT / f"results/scores/{s.model}__{sp}.csv") for sp in s.splits},
        "data_roots": cfg["dataset_roots"], "settings": asdict(s),
        "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for folder in ("src", "lab10")
                          for p in sorted((ROOT / folder).rglob("*.py"))},
        "notebook_sha256": sha256(ROOT / "lab10/explainability.ipynb"),
        "asset_hashes": {p.name: sha256(p) for p in sorted((ROOT / "lab10/assets").iterdir())
                         if p.is_file()},
        "checkpoint_identity_limit": "Hash local; CSV antiguo sin hash: época y scores se contrastan empíricamente.",
    }
    provenance["source_hashes"]["app/inference.py"] = sha256(ROOT / "app/inference.py")
    config_path = ROOT / "configs/xlsr_aasist_frozen_es_omni.yaml"
    if s.model == "C4":
        provenance["run_config_sha256"] = sha256(config_path)
    write_json(OUT / "provenance.json", provenance)
    return model, cfg, datasets, pool, provenance


def action(score: float, thresholds: dict) -> str:
    return ("analista" if score >= thresholds["t2"] else
            "verificar" if score >= thresholds["t1"] else "continuar")


class Predictor:
    """Contabiliza forwards y entradas; tiempos sincronizados, sin carga del disco/modelo."""

    def __init__(self, model, settings: Settings):
        self.model, self.s = model, settings
        self.calls, self.inputs = 0, 0

    def sync(self):
        if self.s.device.startswith("cuda"):
            torch.cuda.synchronize()

    def __call__(self, batch: torch.Tensor) -> np.ndarray:
        values = []
        for start in range(0, len(batch), self.s.batch):
            part = batch[start:start + self.s.batch].to(self.s.device)
            with torch.inference_mode(), torch.autocast(
                device_type=self.s.device.split(":")[0], dtype=torch.bfloat16,
                enabled=self.s.precision == "bfloat16",
            ):
                logits = self.model(part).float()
            values.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            self.calls += 1
            self.inputs += len(part)
        return np.concatenate(values).astype(float)


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-np.logaddexp(0, -x))


def sample_cases(pool: pd.DataFrame, s: Settings, thresholds: dict):
    """Selección descriptiva con scores previos; ningún ajuste de modelo/umbral con test."""
    p = pool.sort_values(["split", "utt_id"]).copy()
    p["prediction_cached"] = (p.cached_score >= thresholds["t2"]).astype(int)
    p["correct_cached"] = p.prediction_cached == p.label
    p["extremeness"] = np.where(p.label == 1, p.cached_score, 1 - p.cached_score)
    chosen = []

    def take(candidates, reason):
        ids = {r["utt_id"] for r in chosen}
        candidates = candidates.loc[~candidates.utt_id.isin(ids)]
        if candidates.empty:
            raise ValueError(f"No hay candidato único para {reason}; agregar otro split de prueba.")
        row = candidates.iloc[0].to_dict()
        row["reason"] = reason
        chosen.append(row)

    take(p[p.correct_cached & (p.extremeness >= 0.99)].sort_values("extremeness", ascending=False),
         "acierto_score_extremo")
    take(p[~p.correct_cached].sort_values("extremeness"), "error")
    p["distance"] = abs(p.cached_score - thresholds["t2"])
    take(p.sort_values("distance"), "cercano_t2")
    subgroup = p[(p.language == "es") | (p.codec.isin(["g711", "gsm", "amr"]))]
    if subgroup.empty:
        # A10 es un subgrupo de ataque de ASVspoof; el escrito previo investigó su silencio.
        subgroup = p[p.attack == "A10"]
    take(subgroup.sort_values(["attack", "utt_id"], ascending=[False, True]), "subgrupo")
    take(p[p.label == 0].sort_values("extremeness", ascending=False), "genuino_distinto")
    local = pd.DataFrame(chosen)
    if local.utt_id.nunique() != 5:
        raise AssertionError("Se requieren cinco IDs únicos.")
    rng = np.random.default_rng(s.seed)
    global_parts = []
    strata = ["split", "label", "attack", "codec"]
    for _, group in p.groupby(strata, sort=True):
        indices = rng.choice(len(group), min(s.per_stratum, len(group)), replace=False)
        global_parts.append(group.iloc[indices])
    global_sample = pd.concat(global_parts).sort_values(["split", "utt_id"]).reset_index(drop=True)
    global_sample.to_csv(OUT / "global_sample.csv", index=False)
    local.to_csv(OUT / "local_selection.csv", index=False)
    required = pd.concat([local, global_sample]).drop_duplicates(["split", "utt_id"])
    required[["split", "utt_id", "dataset", "path", "audio_path"]].to_csv(
        OUT / "required_audio.csv", index=False)
    distribution = global_sample.groupby(strata).size().rename("n").reset_index()
    distribution.to_csv(OUT / "global_distribution.csv", index=False)
    return local, global_sample


def read_wave(row, datasets) -> torch.Tensor:
    return datasets[row["split"]][int(row["dataset_index"])]["wav"]


def reproduce(predict: Predictor, datasets, local, global_sample, thresholds):
    union = pd.concat([local, global_sample]).drop_duplicates(["split", "utt_id"])
    missing = union.loc[~union.audio_path.map(lambda p: Path(p).is_file())]
    if not missing.empty:
        missing[["split", "utt_id", "dataset", "path"]].to_csv(OUT / "missing_audio.csv", index=False)
        raise FileNotFoundError(
            f"Faltan {len(missing)} audios de la muestra predefinida; ver required_audio.csv. "
            "Descargar esos archivos; no sustituir casos por disponibilidad.")
    records, waves = [], {}
    for _, row in union.iterrows():
        wav = read_wave(row, datasets)
        key = (row["split"], row["utt_id"])
        waves[key] = wav
        predict.sync()
        prediction_start = time.perf_counter()
        lo = float(predict(wav[None])[0])
        predict.sync()
        prediction_seconds = time.perf_counter() - prediction_start
        score = float(sigmoid(lo))
        rec = row.to_dict()
        rec.update(log_odds=lo, score=score, prediction=int(score >= thresholds["t2"]),
                   action=action(score, thresholds), t1=thresholds["t1"], t2=thresholds["t2"],
                   score_abs_diff=abs(score - row.cached_score))
        rec["audio_sha256"] = sha256(Path(row.audio_path))
        rec["prediction_seconds"] = prediction_seconds
        rec["correct"] = rec["prediction"] == rec["label"]
        rec["cached_action"] = action(row.cached_score, thresholds)
        records.append(rec)
    verified = pd.DataFrame(records)
    verified.to_csv(OUT / "score_verification.csv", index=False)
    # Los CSV proceden de GPU/bf16; la API/este análisis usan float32. La identidad
    # de pesos ya se verificó por hash y época. Se conserva la tolerancia, se registran
    # desacuerdos y solo se explican las predicciones actuales, sin suponer equivalencia.
    verified["within_historical_atol"] = verified.score_abs_diff <= predict.s.score_atol
    verified.to_csv(OUT / "score_verification.csv", index=False)
    write_json(OUT / "score_agreement.json", {
        "n": len(verified), "atol": predict.s.score_atol,
        "n_outside_atol": int((~verified.within_historical_atol).sum()),
        "max_abs_diff": float(verified.score_abs_diff.max()),
        "action_changes": int((verified.action != verified.cached_action).sum()),
        "historical_runtime": "GPU bfloat16 según results/runs.csv; entorno completo no archivado",
        "current_runtime": f"{predict.s.device}/{predict.s.precision}",
        "interpretation": "Mismo checkpoint por hash/época; CSV no reproducido exactamente. "
            "Se conservan discrepancias, sin ampliar tolerancia ni recalibrar con test. "
            "Cercano_t2 significa seleccionado por mínimo histórico; no mínimo actual de todo eval.",
    })
    ids = local.set_index(["split", "utt_id"])["reason"]
    local_verified = verified.set_index(["split", "utt_id"]).loc[ids.index].reset_index()
    local_verified["reason"] = ids.to_numpy()
    if not local_verified.loc[local_verified.reason == "error", "correct"].eq(False).all():
        raise ValueError("El error candidato no se reproduce; revisar CSV/checkpoint.")
    high = local_verified[local_verified.reason == "acierto_score_extremo"].iloc[0]
    if not high.correct or max(high.score, 1 - high.score) < .99:
        raise ValueError("El acierto de score extremo no se reproduce.")
    local_verified.to_csv(OUT / "local_cases.csv", index=False)
    return local_verified, verified, waves


def precision_audit(predict, verified, waves):
    """Diagnóstico explícito: CPU bf16 vs ejecución actual, sin elegir resultados favorables."""
    if predict.s.device != "cpu" or predict.s.precision != "float32":
        write_json(OUT / "precision_audit_timing.json", {"status": "not_applicable"})
        pd.DataFrame(columns=["utt_id", "cached_score", "score_float32", "score_cpu_bfloat16"]).to_csv(
            OUT / "precision_audit.csv", index=False)
        return
    alternative = Predictor(predict.model, replace(predict.s, precision="bfloat16"))
    selected = verified[(verified.score_abs_diff > predict.s.score_atol) |
                        (verified.get("reason", pd.Series("", index=verified.index)) == "cercano_t2")]
    start = time.perf_counter()
    rows = []
    for row in selected.itertuples():
        score = float(sigmoid(alternative(waves[(row.split, row.utt_id)][None])[0]))
        rows.append({"utt_id": row.utt_id, "split": row.split,
                     "cached_score": row.cached_score, "score_float32": row.score,
                     "score_cpu_bfloat16": score,
                     "abs_diff_cpu_bfloat16": abs(score - row.cached_score)})
    pd.DataFrame(rows).to_csv(OUT / "precision_audit.csv", index=False)
    write_json(OUT / "precision_audit_timing.json", {
        "seconds": time.perf_counter() - start, "inputs": alternative.inputs,
        "forwards": alternative.calls, "includes_model_load": False})


def temporal_starts(n=LENGTH, window=4000, stride=2000):
    """Ventanas completas, incluida una alineada al borde final; no dejar cola sin explicar."""
    starts = list(range(0, max(n - window + 1, 1), stride))
    if starts[-1] != n - window:
        starts.append(max(0, n - window))
    return sorted(set(starts))


def stop_band(wav, lo, hi):
    """La última banda incluye Nyquist (8000 Hz); corrige la exclusión '< hi' original."""
    spec = torch.fft.rfft(wav)
    hz = torch.fft.rfftfreq(wav.shape[-1], 1 / SR)
    keep = (hz >= lo) & ((hz <= hi) if hi == SR // 2 else (hz < hi))
    spec[..., keep] = 0
    return torch.fft.irfft(spec, n=wav.shape[-1])


def smooth_mask(wav, starts, width=4000, ramp=160):
    gain = torch.ones_like(wav)
    for start in starts:
        g = torch.zeros(width)
        edge = .5 * (1 + torch.cos(torch.linspace(0, math.pi, ramp)))
        g[:ramp], g[-ramp:] = edge, edge.flip(0)
        gain[start:start + width] = torch.minimum(gain[start:start + width], g)
    return wav * gain


def explain_local(predict, local, waves):
    s = predict.s
    win, stride = round(s.window_s * SR), round(s.stride_s * SR)
    starts = temporal_starts(window=win, stride=stride)
    temporal, bands, timings, quality = [], [], [], []
    for _, row in local.iterrows():
        wav = waves[(row["split"], row["utt_id"])]
        predict.sync()
        begin, calls, inputs = time.perf_counter(), predict.calls, predict.inputs
        masked = torch.stack([time_mask(wav, start, win) for start in starts])
        delta = row.log_odds - predict(masked)
        banded = torch.stack([stop_band(wav, lo, hi) for lo, hi in BANDS_HZ])
        band_delta = row.log_odds - predict(banded)
        predict.sync()
        timings.append({"kind": "local", "utt_id": row.utt_id,
                        "seconds": time.perf_counter() - begin,
                        "forwards": predict.calls - calls, "inputs": predict.inputs - inputs,
                        "includes_base": False})
        common = {"utt_id": row.utt_id, "split": row.split, "label": row.label,
                  "correct": bool(row.correct), "base_log_odds": row.log_odds}
        for start, d in zip(starts, delta):
            temporal.append({**common, "start_s": start / SR, "end_s": (start + win) / SR,
                             "delta": float(d)})
        for (lo, hi), d in zip(BANDS_HZ, band_delta):
            bands.append({**common, "lo_hz": lo, "hi_hz": hi, "delta": float(d)})
        all_perturbed = torch.cat([masked, banded])
        for i, perturbed in enumerate(all_perturbed):
            original_energy = float(wav.square().mean())
            quality.append({**common, "kind": "time" if i < len(starts) else "band",
                            "perturbation_index": i if i < len(starts) else i - len(starts),
                            "original_rms": math.sqrt(original_energy),
                            "perturbed_rms": float(perturbed.square().mean().sqrt()),
                            "relative_removed_energy": float((wav - perturbed).square().mean()) /
                            max(original_energy, 1e-12),
                            "max_adjacent_jump": float(torch.diff(perturbed).abs().max()),
                            "original_max_jump": float(torch.diff(wav).abs().max()),
                            "clipped_fraction": float((perturbed.abs() >= .999).float().mean())})
        print(f"Local {row.utt_id}: {len(starts)} tramos + {len(BANDS_HZ)} bandas", flush=True)
    frames = [pd.DataFrame(v) for v in (temporal, bands, timings, quality)]
    for frame, name in zip(frames, ("temporal", "local_bands", "local_timings", "perturbation_qc")):
        frame.to_csv(OUT / f"{name}.csv", index=False)
    return frames


def nonoverlap(order, starts, window, k=3):
    chosen = []
    for i in order:
        start = starts[int(i)]
        if all(abs(start - other) >= window for other in chosen):
            chosen.append(start)
        if len(chosen) == k:
            return chosen
    raise ValueError("No se pudieron seleccionar regiones disjuntas.")


def fidelity(predict, local, waves, temporal):
    """Top-3 vs 5 controles aleatorios disjuntos; auditoría descriptiva de sensibilidad."""
    rng = np.random.default_rng(predict.s.seed)
    rows = []
    for _, row in local.iterrows():
        wav = waves[(row["split"], row["utt_id"])]
        table = temporal[(temporal.utt_id == row.utt_id) & (temporal.split == row.split)]
        starts = np.rint(table.start_s.to_numpy() * SR).astype(int).tolist()
        direction = 1 if row.prediction == 1 else -1
        order = np.argsort(-direction * table.delta.to_numpy(), kind="stable")
        top = nonoverlap(order, starts, 4000)
        perturbations, names = [], []
        for label, selected in [("top", top)] + [
            (f"control_{i}", nonoverlap(rng.permutation(len(starts)), starts, 4000))
            for i in range(predict.s.control_repeats)
        ]:
            hard = wav.clone()
            for st in selected:
                hard[st:st + 4000] = 0
            perturbations.extend([hard, smooth_mask(wav, selected)])
            names.extend([(label, "hard", selected), (label, "ramp10ms", selected)])
        scores = predict(torch.stack(perturbations))
        for (label, replacement, selected), logodds in zip(names, scores):
            rows.append({"utt_id": row.utt_id, "split": row.split, "correct": bool(row.correct),
                         "class_direction": direction, "regions": label,
                         "replacement": replacement, "start_samples": json.dumps(selected),
                         "removed_seconds": .75, "base_log_odds": row.log_odds,
                         "perturbed_log_odds": logodds, "delta": row.log_odds - logodds,
                         "directional_drop": direction * (row.log_odds - logodds)})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "fidelity.csv", index=False)
    return frame


def explain_global(predict, global_sample, verified, waves):
    records = []
    predict.sync()
    begin, calls, inputs = time.perf_counter(), predict.calls, predict.inputs
    indexed = verified.set_index(["split", "utt_id"])
    for _, row in global_sample.iterrows():
        key = (row["split"], row["utt_id"])
        wav, base = waves[key], float(indexed.loc[key, "log_odds"])
        deltas = base - predict(torch.stack([stop_band(wav, lo, hi) for lo, hi in BANDS_HZ]))
        for (lo, hi), delta in zip(BANDS_HZ, deltas):
            records.append({"split": row.split, "utt_id": row.utt_id, "label": row.label,
                            "attack": row.attack, "codec": row.codec,
                            "correct": bool(indexed.loc[key, "correct"]),
                            "lo_hz": lo, "hi_hz": hi, "delta": float(delta),
                            "abs_delta": abs(float(delta))})
        print(f"Global {row.split}/{row.utt_id}", flush=True)
    predict.sync()
    cost = {"kind": "global_bands", "n": len(global_sample),
            "seconds": time.perf_counter() - begin,
            "forwards": predict.calls - calls, "inputs": predict.inputs - inputs,
            "includes_base": False}
    if "prediction_seconds" in verified:
        keys = pd.MultiIndex.from_frame(global_sample[["split", "utt_id"]])
        cost["base_prediction_seconds"] = float(indexed.loc[keys, "prediction_seconds"].sum())
        cost["total_seconds_with_base"] = cost["seconds"] + cost["base_prediction_seconds"]
        cost["total_forwards_with_base"] = cost["forwards"] + len(global_sample)
        cost["total_inputs_with_base"] = cost["inputs"] + len(global_sample)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "global_attributions.csv", index=False)
    grouped = frame.groupby(["split", "label", "codec", "lo_hz", "hi_hz"])
    summary = grouped.agg(n=("delta", "size"), mean_delta=("delta", "mean"),
                          sd_delta=("delta", "std"), mean_abs=("abs_delta", "mean"),
                          sd_abs=("abs_delta", "std"), positive_fraction=("delta", lambda a: (a > 0).mean()))
    summary.reset_index().to_csv(OUT / "global_summary.csv", index=False)
    write_json(OUT / "global_timing.json", cost)
    return frame, cost


def benchmark(predict, wav, repeats=5):
    # Calentamiento y repetición: solo inferencia, no extrapolar a ONNX ni otra máquina.
    predict(wav[None])
    durations = []
    for _ in range(repeats):
        predict.sync()
        begin = time.perf_counter()
        predict(wav[None])
        predict.sync()
        durations.append(time.perf_counter() - begin)
    frame = pd.DataFrame({"repeat": range(repeats), "seconds": durations,
                          "kind": "prediction", "inputs": 1, "forwards": 1})
    frame.to_csv(OUT / "prediction_timings.csv", index=False)
    return frame


def risk_probes(predict, local, waves, thresholds):
    """Sondas de silencio/volumen/canal; no confundir sensibilidad con cambio causal de clase."""
    rows = []
    for _, row in local.iterrows():
        wav = waves[(row["split"], row["utt_id"])]
        phone = stop_band(stop_band(wav, 0, 300), 3400, 8000)
        probes = {"volume_half": wav * .5, "volume_double_clipped": (wav * 2).clamp(-1, 1),
                  "phone_band_only": phone, "first_250ms_silenced": time_mask(wav, 0, 4000)}
        # Pares NUEVOS del mismo test: se reutiliza el códec real del proyecto,
        # sin atribuirles scores históricos de asv19_eval_codec ni cambiar protocolos.
        from scripts.precompute_codecs import apply_codec
        for codec in ("g711", "gsm"):
            converted = torch.from_numpy(apply_codec(wav.numpy(), SR, codec))
            if converted.numel() < LENGTH:
                converted = converted.repeat(math.ceil(LENGTH / converted.numel()))
            probes[f"codec_{codec}"] = converted[:LENGTH].clamp(-1, 1)
        logits = predict(torch.stack(list(probes.values())))
        frames = wav[:(LENGTH // 400) * 400].reshape(-1, 400)
        silence = float((frames.square().mean(1).sqrt() < .005).float().mean())
        for name, lo in zip(probes, logits):
            score = float(sigmoid(lo))
            rows.append({"utt_id": row.utt_id, "split": row.split, "label": row.label,
                         "correct": bool(row.correct), "probe": name,
                         "base_score": row.score, "score": score, "delta": row.log_odds - lo,
                         "base_action": row.action, "action": action(score, thresholds),
                         "silence_frame_fraction": silence,
                         "initial_250ms_rms": float(wav[:4000].square().mean().sqrt())})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "risk_probes.csv", index=False)
    return frame


def figures(local, waves, temporal, local_bands, global_attr, fidelity_frame):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(5, 3, figsize=(7.2, 5.86), gridspec_kw={"width_ratios": [2, 2, 1.6]})
    vmax = max(float(temporal.delta.abs().max()), float(local_bands.delta.abs().max()), .01)
    for i, (_, row) in enumerate(local.iterrows()):
        wave = waves[(row["split"], row["utt_id"])]
        table = temporal[(temporal.utt_id == row.utt_id) & (temporal.split == row.split)]
        band = local_bands[(local_bands.utt_id == row.utt_id) & (local_bands.split == row.split)]
        axes[i, 0].plot(np.arange(LENGTH) / SR, wave.numpy(), lw=.45, color="#34534b")
        axes[i, 0].set_title(f"Caso {i+1} · {row.attack} · y={row.label}", fontsize=10, loc="left")
        axes[i, 0].set_ylabel("amplitud")
        centers = (table.start_s + table.end_s) / 2
        colors = np.where(table.delta >= 0, "#b24a3b", "#326c9c")
        axes[i, 1].bar(centers, table.delta, width=.10, color=colors)
        axes[i, 1].set_ylim(-vmax * 1.08, vmax * 1.08)
        axes[i, 1].axhline(0, color="black", lw=.5)
        axes[i, 1].set_ylabel("Δ log-odds")
        axes[i, 2].bar(np.arange(6), band.delta, color=np.where(band.delta >= 0, "#b24a3b", "#326c9c"))
        axes[i, 2].set_ylim(-vmax * 1.08, vmax * 1.08)
        axes[i, 2].axhline(0, color="black", lw=.5)
        axes[i, 2].set_xticks(range(6), ["0–.3", ".3–1", "1–2", "2–3.4", "3.4–5", "5–8"], rotation=45)
        for ax in axes[i]:
            ax.tick_params(labelsize=9)
            ax.yaxis.label.set_size(9)
        if i == 4:
            axes[i, 0].set_xlabel("tiempo (s)")
            axes[i, 1].set_xlabel("centro del segmento (s)")
            axes[i, 2].set_xlabel("banda (kHz)")
    fig.suptitle("Original | silencio brusco 0.25 s | banda FFT anulada\n"
                 "Δ log-odds: rojo > 0; azul < 0", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .94], h_pad=.65)
    fig.savefig(OUT / "local_explanations.png", dpi=240)
    plt.close(fig)
    splits = list(global_attr.split.unique())
    fig, axes = plt.subplots(1, len(splits), figsize=(7.2, 1.9), squeeze=False, sharey=True)
    for ax, split in zip(axes[0], splits):
        subset = global_attr[global_attr.split == split]
        for label, color in [(0, "#326c9c"), (1, "#b24a3b")]:
            group = subset[subset.label == label].groupby("lo_hz", sort=True)
            mean, sd = group.abs_delta.mean(), group.abs_delta.std().fillna(0)
            signed = group.delta.mean()
            x = np.arange(6) + (label - .5) * .28
            ax.errorbar(x, mean, yerr=sd, fmt="o-", capsize=2, color=color,
                        markersize=3, label=f"y={label}, n={subset[subset.label == label].utt_id.nunique()}, |Δ| ± DE")
            ax.plot(x, signed, ":", color=color, alpha=.8)
        ax.axhline(0, color="black", lw=.5)
        ax.set_xticks(range(6), ["0–.3", ".3–1", "1–2", "2–3.4", "3.4–5", "5–8"], rotation=25)
        n = subset[["utt_id"]].drop_duplicates().shape[0]
        ax.set_title(f"{split}: N={n}", fontsize=9)
        ax.set_xlabel("banda (kHz)", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_ylabel("sensibilidad (log-odds)", fontsize=8)
    axes[0, -1].legend(fontsize=7)
    fig.suptitle("Exploratoria · bandas FFT anuladas · punteado: media Δ; puntos: media |Δ| ± DE", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "global_explanations.png", dpi=240)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 1.65))
    for ax, repl in zip(axes, ["hard", "ramp10ms"]):
        sub = fidelity_frame[fidelity_frame.replacement == repl]
        for i, case in enumerate(local.utt_id):
            q = sub[sub.utt_id == case]
            top = q[q.regions == "top"].directional_drop.iloc[0]
            controls = q[q.regions != "top"].directional_drop
            ax.scatter([i], [top], c="#b24a3b", marker="D", label="top-3" if i == 0 else None)
            ax.errorbar([i + .15], [controls.mean()], yerr=[controls.std()], fmt="o",
                        c="#326c9c", capsize=3, label="control ± DE" if i == 0 else None)
        ax.set_title("Brusco (principal)" if repl == "hard" else "Rampa 10 ms (control)")
        ax.set_xticks(range(5), range(1, 6))
        ax.axhline(0, color="black", lw=.5)
        ax.set_xlabel("caso local (mismo orden)")
    axes[0].set_ylabel("caída orientada\n(log-odds)", fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fidelity.png", dpi=220)
    plt.close(fig)


def measured_findings(local, global_attr, fidelity_frame, local_times, prediction_times, global_cost):
    hard = fidelity_frame[fidelity_frame.replacement == "hard"]
    top = hard[hard.regions == "top"].set_index("utt_id").directional_drop
    controls = hard[hard.regions != "top"].groupby("utt_id").directional_drop.mean()
    smooth = fidelity_frame[(fidelity_frame.replacement == "ramp10ms") &
                            (fidelity_frame.regions == "top")].set_index("utt_id").directional_drop
    top, controls = top.align(controls)
    smooth = smooth.reindex(top.index)
    bands = global_attr.groupby("lo_hz").abs_delta.mean()
    outside = global_attr[(global_attr.lo_hz < 300) | (global_attr.lo_hz >= 3400)]
    findings = {
        "local_n": len(local), "local_errors": local.loc[~local.correct, "utt_id"].tolist(),
        "global_n": len(global_attr[["split", "utt_id"]].drop_duplicates()),
        "global_sampling": "igual cantidad por split/clase/ataque/códec; descriptivo, no prevalencia natural",
        "fidelity_top_beats_control": int((top > controls).sum()),
        "fidelity_n": len(top), "fidelity_mean_gap": float((top - controls).mean()),
        "ramp_mean_absolute_change": float((top - smooth).abs().mean()),
        "outside_phone_mean_abs": float(outside.abs_delta.mean()),
        "largest_mean_abs_band_lo": int(bands.idxmax()),
        "prediction_seconds_median": float(prediction_times.seconds.median()),
        "local_seconds_median": float(local_times.seconds.median()),
        "global_seconds": global_cost["seconds"],
        "global_total_seconds_with_base": global_cost.get("total_seconds_with_base", global_cost["seconds"]),
        "global_total_forwards_with_base": global_cost.get("total_forwards_with_base", global_cost["forwards"]),
        "global_forwards": global_cost["forwards"], "global_inputs": global_cost["inputs"],
        "local_inputs_without_base": int(local_times.inputs.iloc[0]),
        "score_action_changes_against_cached": int((local.action != local.cached_action).sum()),
    }
    write_json(OUT / "findings.json", findings)
    return findings


def export_parameters(s, provenance, local, global_sample):
    values = [
        ("Modelo/checkpoint", f"runs/{s.model}/best.pt; SHA256 {provenance['checkpoint_sha256'][:12]}…",
         "versión indicada; no reentrenar"),
        ("Datos / muestra", f"{', '.join(s.splits)}; semilla 42; 5 locales; N={len(global_sample)} globales",
         "test; dirigidas / estratos"),
        ("Unidad/clase", "64600 muestras; 16 kHz; 4.0375 s; y=1 sintética; mono, clamp [-1,1]; recortar/repetir",
         "mismo AudioDataset; audio corto y borde"),
        ("Score/umbrales", (f"Δ log-odds; t1={provenance['thresholds']['t1']:.10g}; "
         f"t2={provenance['thresholds']['t2']:.10g}"), "umbrales de negocio sobre softmax crudo"),
        ("Tiempo", "ancho 0.25 s; paso 0.125 s; 32 posiciones; última alineada al final", "resolución/costo; cubrir borde"),
        ("Bandas (Hz)", "0–300; 300–1000; 1000–2000; 2000–3400; 3400–5000; 5000–8000", "comparar dentro/fuera de telefonía; incluir Nyquist"),
        ("Principal, Figs. 1–2", f"silencio brusco (tiempo), FFT anulada (bandas); B={s.batch}; {s.device}/{s.precision}",
         "Nyquist incluido; eval(), sin augmentation"),
        ("Control, Fig. 3", "rampa de 10 ms frente a máscara brusca; top-3 y 5 controles/caso",
         "solo sensibilidad al reemplazo; no genera Figs. 1–2"),
    ]
    table = pd.DataFrame(values, columns=["parámetro", "valor", "razón"])
    table.to_csv(OUT / "parameters.csv", index=False)
    return table
