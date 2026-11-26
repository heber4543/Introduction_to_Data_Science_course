"""Paso 10. Muestra de metricas para el analisis de redundancia (covarianza) en Colab.

Regenera, con las mismas semillas del dataset (plan_seed y sim_seed de la corrida),
las 59 metricas de n agentes elegidos al azar entre los 713 millones. Como el azar
depende solo de la semilla, del agente y del ensayo, las metricas son identicas a
las que vio la red. No lee los archivos grandes de LUSTRE.

Guarda runs/.../ann/<red>/muestra_metricas.npz con
  X           (n, 59) metricas en float16, igual que el dataset
  names       nombres de las 59 metricas
  agente      indice global de cada agente
  params      (n, 5) parametros verdaderos alpha, phi, rho, epsilon, K
  division    0 entrenamiento, 1 validacion, 2 prueba (misma division de la red)

Corre en el ambiente rlwm-sim (numba).
Uso:  python -m rlwm_sim.muestra_metricas configs/eval_soluciones.json [--n 200000] [--seed 20261129]
"""
import argparse
import json
import time

import numpy as np

from .eval_soluciones import PARAMS, _expand, split_of_points
from .grid import Grid
from .random_env import RandomSequences


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20261129)
    args = ap.parse_args(argv)

    cfg = json.load(open(args.config, encoding="utf-8"))
    ann = json.load(open(cfg["ann_config"], encoding="utf-8"))
    sim = json.load(open(ann["sim_config"], encoding="utf-8"))
    model_dir = _expand(sim["run_dir"]) / "ann" / ann["ann_run_name"]

    grid = Grid(sim["grid"])
    R = int(sim["replicates"])
    N = grid.n_total * R
    seq = RandomSequences(sim)

    rng = np.random.default_rng(args.seed)
    cand = np.unique(rng.integers(0, N, size=args.n + args.n // 100 + 10, dtype=np.int64))
    agents = np.sort(rng.permutation(cand)[:args.n])
    gidx = agents // R
    th = grid.thetas(gidx)
    print(f"{len(agents):,} agentes al azar de {N:,}", flush=True)

    t0 = time.time()
    _, X = seq.simulate_features(th, agents.astype(np.uint64))
    print(f"Metricas regeneradas en {time.time() - t0:.0f} s", flush=True)

    sp = ann["split"]
    division = split_of_points(gidx, ann["seed"], sp["train"], sp["val"])
    out = model_dir / "muestra_metricas.npz"
    np.savez_compressed(out, X=X.astype(np.float16), names=np.array(seq.names()), agente=agents,
                        params=th[:, :5].astype(np.float32), param_names=np.array(PARAMS),
                        division=division)
    print(f"Guardado en {out}", flush=True)
    print(f"Por division: entrenamiento {int((division == 0).sum()):,}, "
          f"validacion {int((division == 1).sum()):,}, prueba {int((division == 2).sum()):,}", flush=True)
    print(f"Columnas con algun NaN: {int(np.isnan(X).any(axis=0).sum())} de {X.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
