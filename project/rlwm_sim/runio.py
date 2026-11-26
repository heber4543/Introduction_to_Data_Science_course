"""Utilidades de corrida: configuracion, huella del codigo y manifiesto."""
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["run_dir"] = os.path.expandvars(os.path.expanduser(cfg["run_dir"]))
    return cfg


def code_hash():
    """Huella (SHA-256) de los archivos .py del paquete. Si cambias el codigo,
    cambia la huella; asi se detecta una corrida armada con versiones mezcladas."""
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def versions():
    import numpy, numba
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "numba": numba.__version__, "platform": platform.platform()}


def run_paths(cfg):
    rd = Path(cfg["run_dir"])
    return {"run_dir": rd, "shards": rd / "shards", "logs": rd / "logs",
            "manifest": rd / "manifest.json", "config": rd / "config.json",
            "plan_npz": rd / "plan.npz", "plan_json": rd / "plan.json"}


def read_manifest(cfg):
    with open(run_paths(cfg)["manifest"], encoding="utf-8") as f:
        return json.load(f)


def set_threads():
    """Usa tantos hilos como nucleos asigno SLURM (variable SLURM_CPUS_PER_TASK)."""
    import numba
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n:
        numba.set_num_threads(min(int(n), numba.config.NUMBA_NUM_THREADS))
    return numba.get_num_threads()
