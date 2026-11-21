"""Ejecuta desde kernel limpio con el MISMO Python que invoca este módulo."""

import json
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from lab10.audit import OUT, ROOT, sha256, write_json


class ProgressClient(NotebookClient):
    """Muestra progreso de audio/etapas sin volcar tablas ni imágenes al terminal."""

    def process_message(self, msg, cell, cell_index):
        if msg.get("msg_type") == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)


def run():
    notebook = nbformat.read(ROOT / "lab10/explainability.ipynb", as_version=4)
    nbformat.validate(notebook)
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "execution_status.json", {"status": "RUNNING"})
    with tempfile.TemporaryDirectory(prefix="lab10-kernel-") as folder:
        kernel = Path(folder) / "lab10"
        kernel.mkdir()
        (kernel / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Lab10 (entorno actual)", "language": "python",
        }))
        manager = KernelManager(kernel_name="lab10",
                                kernel_spec_manager=KernelSpecManager(kernel_dirs=[folder]))
        client = ProgressClient(notebook, km=manager, timeout=7200,
                                resources={"metadata": {"path": str(ROOT)}})
        started = time.perf_counter()

        def cell_start(cell, cell_index, **kwargs):
            if cell.cell_type == "code":
                print(f"Celda {cell_index} ({', '.join(cell.metadata.get('tags', []))}): "
                      f"inicio; {time.perf_counter() - started:.1f} s acumulados", flush=True)

        def cell_executed(cell, cell_index, **kwargs):
            print(f"Celda {cell_index}: completada; "
                  f"{time.perf_counter() - started:.1f} s acumulados", flush=True)

        client.on_cell_start = cell_start
        client.on_cell_executed = cell_executed
        try:
            client.execute()
        except Exception as exc:
            write_json(OUT / "execution_status.json", {
                "status": "BLOCKED_OR_FAILED", "error_type": type(exc).__name__,
                "message": str(exc),
            })
            print("No se completó la ejecución. Ver lab10/outputs/execution_status.json.", file=sys.stderr)
            raise
    executed = OUT / "explainability.executed.ipynb"
    nbformat.write(notebook, executed)
    outputs = {p.name: sha256(p) for p in sorted(OUT.iterdir())
               if p.suffix in {".csv", ".png", ".json", ".ipynb"}
               and p.name not in {"artifact_manifest.json", "execution_status.json"}}
    write_json(OUT / "artifact_manifest.json", {"outputs": outputs})
    write_json(OUT / "execution_status.json", {"status": "NOTEBOOK_EXECUTED"})
    print(f"Notebook ejecutado: {executed}")
    print(f"Tablas y figuras: {OUT}")


if __name__ == "__main__":
    run()
