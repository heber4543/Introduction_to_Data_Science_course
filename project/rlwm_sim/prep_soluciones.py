"""Paso 9a de la evaluacion de soluciones multiples (D-073).

Corre con el Python base del cluster (module load python/miniforge), que trae
PyTorch y pandas. No importa numba ni otros modulos del simulador.

Hace dos cosas:
1. Exporta la red entrenada (modelo.pt) a numpy (modelo_numpy.npz), para que el
   paso 9b la use dentro del ambiente rlwm-sim, que tiene numba pero no PyTorch.
2. Prepara los datos reales de fase 3: rellena respuestas y ensayos faltantes,
   arma la secuencia (plan) de cada participante y la junta con sus parametros
   de maxima verosimilitud (MLE) de la tesis. Guarda real/resultados/datos_reales.npz
   con permisos solo para el usuario.

Relleno acordado con el usuario:
- Respuesta faltante: la ultima respuesta que dio el participante a ese mismo
  estimulo. Si falta en la primera presentacion, la siguiente respuesta.
- Ensayo faltante (estimulo con menos presentaciones de las esperadas): se agrega
  al final de su bloque, con la ultima respuesta a ese estimulo.

Por proteccion de datos, el script se detiene si el archivo de ensayos trae
columnas distintas a las 8 que usa (por ejemplo nombre, usuario o edad).

Uso:  python -m rlwm_sim.prep_soluciones configs/eval_soluciones.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PARAMS = ("alpha", "phi", "rho", "epsilon", "K")
COL_KEYS = ("user", "phase", "trial", "block", "stim", "choice", "correct", "set_size")


def _expand(p):
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


def model_dir_of(cfg):
    ann = json.load(open(cfg["ann_config"], encoding="utf-8"))
    sim = json.load(open(ann["sim_config"], encoding="utf-8"))
    return _expand(sim["run_dir"]) / "ann" / ann["ann_run_name"]


# ---------------------------------------------------------------- 1. red a numpy
def export_model(model_dir):
    ck = torch.load(model_dir / "modelo.pt", map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("net.") and k.endswith(".weight")})
    arrays = {}
    for j, i in enumerate(layers):
        arrays[f"W{j}"] = sd[f"net.{i}.weight"].detach().cpu().numpy().astype(np.float32)
        arrays[f"b{j}"] = sd[f"net.{i}.bias"].detach().cpu().numpy().astype(np.float32)
    out = model_dir / "modelo_numpy.npz"
    np.savez(out,
             n_layers=np.int64(len(layers)),
             selected=np.asarray(ck["selected"], dtype=np.int64),
             names=np.array([str(n) for n in ck["names"]]),
             median=np.asarray(ck["median"], dtype=np.float32),
             mean=np.asarray(ck["mean"], dtype=np.float32),
             std=np.asarray(ck["std"], dtype=np.float32),
             target_offset=np.asarray(ck["target_offset"], dtype=np.float32),
             target_scale=np.asarray(ck["target_scale"], dtype=np.float32),
             targets=np.array([str(t) for t in ck["targets"]]),
             **arrays)
    sizes = [arrays["W0"].shape[1]] + [arrays[f"W{j}"].shape[0] for j in range(len(layers))]
    print(f"Red exportada a {out} (neuronas por capa {' -> '.join(map(str, sizes))}, "
          f"{len(ck['selected'])} entradas seleccionadas)", flush=True)


# ---------------------------------------------------------------- 2. datos reales
def impute(d, c, presentations):
    """Rellena respuestas vacias y agrega presentaciones faltantes al final del bloque."""
    key = [c["user"], c["block"], c["stim"]]
    d = d.sort_values([c["user"], c["trial"]], kind="stable").reset_index(drop=True)

    missing = d[c["choice"]].isna()
    d[c["choice"]] = d.groupby(key, sort=False)[c["choice"]].ffill()   # ultima respuesta a ese estimulo
    d[c["choice"]] = d.groupby(key, sort=False)[c["choice"]].bfill()   # si era la primera presentacion, la siguiente
    n_imp = missing.groupby(d[c["user"]]).sum().astype(int).to_dict()
    print(f"Respuestas vacias rellenadas: {int(missing.sum())}", flush=True)
    left = int(d[c["choice"]].isna().sum())
    if left:
        print(f"Aviso: {left} respuestas siguen vacias (estimulo sin ninguna respuesta); cuentan como error",
              flush=True)

    block_end = d.groupby([c["user"], c["block"]])[c["trial"]].max().to_dict()
    added, n_add, k_in_block = [], {}, {}
    for (u, b, s), g in d.groupby(key, sort=False):
        k = presentations - len(g)
        if k < 0:
            print(f"Aviso: participante {u}, bloque {b}: un estimulo tiene {len(g)} presentaciones "
                  f"(se esperaban {presentations}); se deja como esta", flush=True)
        if k <= 0:
            continue
        for _ in range(k):
            j = k_in_block.get((u, b), 0)
            k_in_block[(u, b)] = j + 1
            row = g.iloc[-1].copy()                                  # misma respuesta y respuesta correcta
            row[c["trial"]] = block_end[(u, b)] + 0.5 + 0.001 * j    # justo despues del ultimo ensayo del bloque
            added.append(row)
            n_add[u] = n_add.get(u, 0) + 1
        print(f"Participante {u}, bloque {b}: se agregan {k} ensayo(s) al final del bloque "
              f"(estimulo con {len(g)} de {presentations} presentaciones; respuesta = su ultima respuesta "
              f"a ese estimulo)", flush=True)
    if added:
        d = pd.concat([d, pd.DataFrame(added)], ignore_index=True)
        d[c["trial"]] = d[c["trial"]].astype(np.float64)
        d[c["choice"]] = d[c["choice"]].astype(np.float64)
        d = d.sort_values([c["user"], c["trial"]], kind="stable").reset_index(drop=True)
    return d, n_imp, n_add


def build_plan(g, c, base):
    """Secuencia (plan) y elecciones de un participante, en el formato del simulador."""
    keys = ("new_block", "block", "stim", "ns", "correct", "iteration")
    cols = {k: [] for k in keys}
    choices = []
    for b_i, b in enumerate(pd.unique(g[c["block"]])):
        db = g[g[c["block"]] == b]
        stims = list(pd.unique(db[c["stim"]]))
        smap = {s: i for i, s in enumerate(stims)}
        if c.get("set_size") and int(db[c["set_size"]].iloc[0]) != len(stims):
            print(f"Aviso: bloque {b} con {len(stims)} estimulos distintos pero set_size "
                  f"{int(db[c['set_size']].iloc[0])}", flush=True)
        it = {}
        for j, (s, a, cor) in enumerate(zip(db[c["stim"]], db[c["choice"]], db[c["correct"]])):
            it[s] = it.get(s, 0) + 1
            cols["new_block"].append(j == 0)
            cols["block"].append(b_i)
            cols["stim"].append(smap[s])
            cols["ns"].append(len(stims))
            cols["correct"].append(int(cor) - base)
            cols["iteration"].append(it[s])
            choices.append(-1 if pd.isna(a) else int(a) - base)
    return {k: np.array(v) for k, v in cols.items()}, np.array(choices)


def prepare_real(cfg):
    rc = cfg["real"]
    c = rc["cols"]
    need = [c[k] for k in COL_KEYS]
    d = pd.read_parquet(_expand(rc["trials_file"]))
    extra = [x for x in d.columns if x not in need]
    if extra:
        sys.exit(f"El archivo de ensayos trae columnas que este analisis no usa: {extra}.\n"
                 f"Por proteccion de datos usa la copia minima, con solo {need}.")
    lack = [x for x in need if x not in d.columns]
    if lack:
        sys.exit(f"Faltan columnas en el archivo de ensayos: {lack}. Existen: {list(d.columns)}")
    d = d[d[c["phase"]] == rc["phase"]].copy()
    if d.empty:
        sys.exit(f"No hay ensayos con {c['phase']} = {rc['phase']}.")
    d[c["trial"]] = d[c["trial"]].astype(np.float64)

    d, n_imp, n_add = impute(d, c, int(rc["presentations"]))
    base = int(rc.get("answer_base", 1))
    users, plans, chs = [], [], []
    for u, g in d.groupby(c["user"], sort=True):
        plan, ch = build_plan(g, c, base)
        users.append(u)
        plans.append(plan)
        chs.append(ch)
    lens = sorted({len(ch) for ch in chs})
    if len(lens) != 1:
        sys.exit(f"Tras el relleno, los participantes tienen distinto numero de ensayos: {lens}")

    P = pd.read_parquet(_expand(rc["params_file"]))
    P = P[P[rc["params_phase_col"]] == rc["phase"]].set_index(rc["params_user_col"])
    if not P.index.is_unique:
        sys.exit("El archivo de parametros tiene mas de una fila por participante en esta fase.")
    miss = [u for u in users if u not in P.index]
    if miss:
        sys.exit(f"Participantes sin parametros MLE en fase {rc['phase']}: {miss}")
    mle = np.array([[float(P.at[u, p]) for p in PARAMS] for u in users], dtype=np.float64)

    out_dir = _expand(rc["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o700)
    out = out_dir / "datos_reales.npz"

    def stack(k, dt):
        return np.stack([p[k] for p in plans]).astype(dt)

    np.savez(out,
             user_id=np.array(users, dtype=np.float64),
             new_block=stack("new_block", np.bool_),
             block=stack("block", np.int16),
             stim=stack("stim", np.int8),
             ns=stack("ns", np.int8),
             correct=stack("correct", np.int8),
             iteration=stack("iteration", np.int16),
             choices=np.stack(chs).astype(np.int16),
             mle=mle,
             param_names=np.array(PARAMS),
             n_imputed=np.array([n_imp.get(u, 0) for u in users], dtype=np.int64),
             n_added=np.array([n_add.get(u, 0) for u in users], dtype=np.int64))
    os.chmod(out, 0o600)
    print(f"Datos reales listos: {len(users)} participantes x {lens[0]} ensayos -> {out}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args(argv)
    cfg = json.load(open(args.config, encoding="utf-8"))
    export_model(model_dir_of(cfg))
    prepare_real(cfg)


if __name__ == "__main__":
    main()
