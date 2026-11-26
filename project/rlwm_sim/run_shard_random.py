"""Paso 2b. Simula un fragmento con secuencia aleatoria por agente y calcula
sus metricas en la misma tarea.

Uso:  python -m rlwm_sim.run_shard_random configs/fase3_rejilla001_secaleatoria.json --shard 7
      (sin --shard se toma SLURM_ARRAY_TASK_ID)

Salida:
  shards/shard_00007.npz      agent_start, agent_end, replicates, codes, replica, choices (n, 90) uint8
  features/features_00007.npz agent_start, agent_end, replicates, codes, replica, X (n, 59) float16
  features/feature_names.json nombres de las columnas de X
Si ambos archivos del fragmento ya existen, se omite.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

from .grid import Grid, agent_to_grid
from .random_env import RandomSequences
from .runio import load_config, run_paths, read_manifest, code_hash, set_threads, PKG_DIR

CHUNK = 250_000


def _atomic_savez(path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--allow-code-change", action="store_true")
    args = ap.parse_args(argv)
    shard = args.shard if args.shard is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])

    cfg = load_config(args.config)
    if cfg.get("sequence", {}).get("mode") != "random_per_agent":
        sys.exit("Esta configuracion no usa secuencia aleatoria por agente; usa run_shard.")
    paths = run_paths(cfg)
    man = read_manifest(cfg)
    if man["code_hash"] != code_hash() and not args.allow_code_change:
        sys.exit("El codigo cambio desde prepare_run (code_hash distinto). Prepara una corrida nueva "
                 "o usa --allow-code-change.")
    fdir = paths["run_dir"] / "features"
    fdir.mkdir(exist_ok=True)
    seq = RandomSequences(cfg)
    names = seq.names()
    names_file = fdir / "feature_names.json"
    if not names_file.exists():
        tmp = fdir / f"feature_names.tmp{os.getpid()}.json"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"names": names, "dtype": "float16", "sequence": cfg["sequence"],
                       "features_code_hash": hashlib.sha256((PKG_DIR / "features.py").read_bytes()).hexdigest()[:16]},
                      f, indent=2, ensure_ascii=False)
        os.replace(tmp, names_file)

    out_c = paths["shards"] / f"shard_{shard:05d}.npz"
    out_f = fdir / f"features_{shard:05d}.npz"
    if out_c.exists() and out_f.exists():
        print(f"fragmento {shard} ya existe; se omite.")
        return 0

    grid = Grid(cfg["grid"])
    R = int(man.get("replicates", 1))
    n_shards = man["n_shards"]
    if not 0 <= shard < n_shards:
        sys.exit(f"fragmento {shard} fuera de rango 0..{n_shards - 1}")
    threads = set_threads()
    start, end = grid.shard_bounds(shard, n_shards, R)
    n = end - start
    choices = np.empty((n, seq.T), dtype=np.uint8)
    X = np.empty((n, len(names)), dtype=np.float16)
    codes = np.empty((n, len(grid.axes)), dtype=grid.code_dtype())
    replica = np.empty(n, dtype=np.uint16)

    t0 = time.time()
    for a in range(start, end, CHUNK):
        b = min(a + CHUNK, end)
        idx = np.arange(a, b, dtype=np.int64)
        gidx, rep = agent_to_grid(idx, R)
        ch, Xf = seq.simulate_features(grid.thetas(gidx), idx.astype(np.uint64))
        choices[a - start:b - start] = ch
        X[a - start:b - start] = Xf
        codes[a - start:b - start] = grid.codes(gidx)
        replica[a - start:b - start] = rep
    t_calc = time.time() - t0

    t1 = time.time()
    meta = dict(agent_start=start, agent_end=end, replicates=R, codes=codes, replica=replica)
    _atomic_savez(out_c, choices=choices, **meta)
    _atomic_savez(out_f, X=X, **meta)
    info = {"shard": shard, "agent_start": start, "agent_end": end, "n_agents": n,
            "seconds_compute": round(t_calc, 2), "seconds_write": round(time.time() - t1, 2),
            "threads": threads, "host": os.uname().nodename, "job_id": os.environ.get("SLURM_JOB_ID")}
    with open(paths["logs"] / f"shard_{shard:05d}.json", "w") as f:
        json.dump(info, f)
    print(json.dumps(info))
    return 0


if __name__ == "__main__":
    sys.exit(main())
