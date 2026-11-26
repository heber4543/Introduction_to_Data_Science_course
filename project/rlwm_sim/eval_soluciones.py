"""Paso 9b de la evaluacion de soluciones multiples (D-073).

Corre en el ambiente rlwm-sim (numba). Necesita lo que deja el paso 9a
(prep_soluciones): la red en numpy (modelo_numpy.npz) y real/resultados/datos_reales.npz.

Pregunta: dos conjuntos de parametros distintos, en la misma secuencia de
estimulos, ¿generan decisiones igual de parecidas a las decisiones observadas?

Parte A (participantes reales, fase 3). Para cada participante se comparan los
parametros MLE de la tesis (techo) y los que predice la red a partir de sus 59
metricas. Con cada conjunto se simulan n_sims agentes en la secuencia real del
participante y se compara, ensayo por ensayo, cada secuencia simulada con la real.

Parte B (agentes simulados de prueba, verdad conocida). Se eligen n_agents al azar
del conjunto de prueba de la red (division 60/20/20 por punto de la rejilla). Sus
decisiones y metricas se regeneran exactamente con las mismas semillas del dataset.
El techo son sus parametros verdaderos.

Medidas por conjunto de parametros (promedio y DE entre las n_sims simulaciones):
- coinc: proporcion de ensayos en que la decision simulada es igual a la observada.
- coinc_err: lo mismo, solo en los ensayos en que la persona (o agente) se equivoco.
- acc_sim: precision de los agentes simulados.
- nll: verosimilitud negativa de las decisiones observadas (solo en la tabla).
Relativas al techo: coinc_rel = coinc(red) / coinc(techo), igual para errores,
y banderas para cada umbral (por defecto 80, 90 y 95 %).

Las simulaciones de los dos conjuntos usan los mismos numeros aleatorios
(mismos indices de agente y semilla). Asi la diferencia entre ambos se debe solo
a los parametros y no a la suerte (numeros aleatorios comunes).

Parametros de la red: K se redondea al entero mas cercano y todos se recortan al
rango de entrenamiento (seccion "clip" de la configuracion).

Uso:  python -m rlwm_sim.eval_soluciones configs/eval_soluciones.json [--parte ambas|reales|simulados]
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from .features import compute_features, feature_names
from .grid import Grid
from .model import PARAM_NAMES, nll, simulate
from .random_env import RandomSequences

PARAMS = ("alpha", "phi", "rho", "epsilon", "K")


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _expand(p):
    return Path(os.path.expandvars(os.path.expanduser(str(p))))


# ---------------------------------------------------------------- division (identica a train_ann.py)
def _splitmix64(x):
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def split_of_points(gidx, seed, f_train, f_val):
    """0 = entrenamiento, 1 = validacion, 2 = prueba, para cada indice de punto."""
    h = _splitmix64(np.asarray(gidx, dtype=np.uint64) ^ np.uint64(seed))
    u = (h >> np.uint64(11)).astype(np.float64) * (1.0 / 2 ** 53)
    return np.where(u < f_train, 0, np.where(u < f_train + f_val, 1, 2)).astype(np.int8)


# ---------------------------------------------------------------- red en numpy
class NumpyNet:
    """La misma red de train_ann.py (capas lineales con ReLU), aplicada con numpy."""

    def __init__(self, path):
        z = np.load(path)
        self.n = int(z["n_layers"])
        self.W = [z[f"W{j}"].astype(np.float64) for j in range(self.n)]
        self.b = [z[f"b{j}"].astype(np.float64) for j in range(self.n)]
        self.sel = z["selected"].astype(np.int64)
        self.names = [str(x) for x in z["names"]]
        self.med = z["median"].astype(np.float64)
        self.mean = z["mean"].astype(np.float64)
        self.std = z["std"].astype(np.float64)
        self.off = z["target_offset"].astype(np.float64)
        self.scale = z["target_scale"].astype(np.float64)
        self.targets = tuple(str(x) for x in z["targets"])

    def predict(self, X):
        """(n, 59) metricas -> (n, 5) parametros en su escala original, sin recortar."""
        x = np.asarray(X, dtype=np.float32).astype(np.float16).astype(np.float64)  # como en el entrenamiento
        x = np.where(np.isnan(x), self.med, x)
        x = (x - self.mean) / self.std
        x = x[:, self.sel]
        for j in range(self.n):
            x = x @ self.W[j].T + self.b[j]
            if j < self.n - 1:
                x = np.maximum(x, 0.0)
        return x * self.scale + self.off


def clip_params(raw, clip):
    p = np.array(raw, dtype=np.float64, copy=True)
    for j, name in enumerate(PARAMS):
        lo, hi = clip[name]
        if name == "K":
            p[:, j] = np.rint(p[:, j])
        p[:, j] = np.clip(p[:, j], lo, hi)
    return p


def thetas_from(p5, fixed):
    p5 = np.atleast_2d(p5)
    th = np.empty((p5.shape[0], len(PARAM_NAMES)), dtype=np.float64)
    th[:, :5] = p5
    for k in ("beta", "bias_wm", "bias_rl"):
        th[:, PARAM_NAMES.index(k)] = fixed[k]
    return th


# ---------------------------------------------------------------- medidas
def measures(theta, plan, choices, n_sims, seed, id_base):
    """Simula n_sims agentes con theta en el plan y compara con las decisiones observadas."""
    ids = np.arange(n_sims, dtype=np.uint64) + np.uint64(id_base)
    S = simulate(np.repeat(theta[None, :], n_sims, axis=0), ids, seed, plan)
    r = np.asarray(choices, dtype=np.int16)
    c = np.asarray(plan["correct"], dtype=np.int16)
    valid = r >= 0
    Si = S.astype(np.int16)
    coinc = (Si[:, valid] == r[valid]).mean(axis=1)
    err = valid & (r != c)
    acc = (Si == c).mean(axis=1)
    out = {"coinc_mean": float(coinc.mean()), "coinc_sd": float(coinc.std(ddof=1))}
    if err.any():
        ce = (Si[:, err] == r[err]).mean(axis=1)
        out["coinc_err_mean"] = float(ce.mean())
        out["coinc_err_sd"] = float(ce.std(ddof=1))
    else:
        out["coinc_err_mean"] = float("nan")
        out["coinc_err_sd"] = float("nan")
    out["acc_sim_mean"] = float(acc.mean())
    out["nll"] = float(nll(theta[None, :], r[None, :], plan)[0])
    return out, S


def _ratio(a, b):
    return a / b if (b > 0 and not np.isnan(a)) else float("nan")


def compare(ref, ref_p5, red_raw5, red_p5, th_ref, th_red, plan, choices, n_sims, seed, id_base,
            thresholds):
    """Una fila de resultados: techo (ref) contra red, con los mismos numeros aleatorios."""
    row = {}
    for j, p in enumerate(PARAMS):
        row[f"{p}_{ref}"] = float(ref_p5[j])
        row[f"{p}_red"] = float(red_p5[j])
        row[f"{p}_red_sin_recorte"] = float(red_raw5[j])
        row[f"dif_{p}"] = float(red_p5[j] - ref_p5[j])
    r = np.asarray(choices, dtype=np.int16)
    c = np.asarray(plan["correct"], dtype=np.int16)
    valid = r >= 0
    row["n_ensayos"] = int(valid.sum())
    row["acc_observada"] = float((r[valid] == c[valid]).mean())
    row["n_errores_observados"] = int((valid & (r != c)).sum())
    m_ref, S_ref = measures(th_ref, plan, r, n_sims, seed, id_base)
    m_red, S_red = measures(th_red, plan, r, n_sims, seed, id_base)
    for k, v in m_ref.items():
        row[f"{k}_{ref}"] = v
    for k, v in m_red.items():
        row[f"{k}_red"] = v
    row["coinc_rel"] = _ratio(m_red["coinc_mean"], m_ref["coinc_mean"])
    row["coinc_err_rel"] = _ratio(m_red["coinc_err_mean"], m_ref["coinc_err_mean"])
    for th in thresholds:
        tag = int(round(th * 100))
        for k in ("coinc_rel", "coinc_err_rel"):
            row[f"{k}_{tag}"] = "" if np.isnan(row[k]) else int(row[k] >= th)
    return row, S_ref, S_red


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows, ref, thresholds, title):
    def col(k):
        return np.array([np.nan if r[k] == "" else r[k] for r in rows], dtype=np.float64)

    def q(x):
        x = x[~np.isnan(x)]
        if x.size == 0:
            return "sin datos"
        return (f"mediana {np.median(x):.3f} [RIC {np.percentile(x, 25):.3f} a "
                f"{np.percentile(x, 75):.3f}] (n={x.size})")

    log(title)
    log(f"  Coincidencia con {ref}:          {q(col(f'coinc_mean_{ref}'))}")
    log(f"  Coincidencia con red:          {q(col('coinc_mean_red'))}")
    log(f"  Relativa al techo:             {q(col('coinc_rel'))}")
    log(f"  En errores con {ref}:           {q(col(f'coinc_err_mean_{ref}'))}")
    log(f"  En errores con red:            {q(col('coinc_err_mean_red'))}")
    log(f"  En errores relativa al techo:  {q(col('coinc_err_rel'))}")
    for k, name in (("coinc_rel", "coincidencia"), ("coinc_err_rel", "coincidencia en errores")):
        x = col(k)
        x = x[~np.isnan(x)]
        txt = ", ".join(f">= {int(round(th * 100))}% del techo: {int((x >= th).sum())}/{x.size}"
                        for th in thresholds)
        log(f"  {name}: {txt}")
    for p in PARAMS:
        log(f"  |diferencia en {p}| (red - {ref}): {q(np.abs(col(f'dif_{p}')))}")


# ---------------------------------------------------------------- parte A
def part_real(cfg, net, fixed):
    rc = cfg["real"]
    out_dir = _expand(rc["out_dir"])
    with np.load(out_dir / "datos_reales.npz") as z:
        D = {k: z[k] for k in z.files}
    users = D["user_id"]
    n_p, T = D["choices"].shape
    n_sims, seed, ths = int(cfg["n_sims"]), int(cfg["seed"]), cfg["thresholds"]
    plans = []
    for i in range(n_p):
        plans.append({"new_block": D["new_block"][i], "block": D["block"][i], "stim": D["stim"][i],
                      "ns": D["ns"][i], "correct": D["correct"][i], "iteration": D["iteration"][i],
                      "n_actions": 3, "max_ns": int(D["ns"][i].max())})
        if feature_names(plans[i]) != net.names:
            sys.exit(f"Las metricas del participante {users[i]} no coinciden con las de la red "
                     "(tamanos de conjunto distintos).")
    log(f"Parte A: {n_p} participantes x {T} ensayos, {n_sims} simulaciones por conjunto de parametros")

    X = np.vstack([compute_features(D["choices"][i], plans[i]) for i in range(n_p)])
    raw = net.predict(X)
    red = clip_params(raw, cfg["clip"])
    mle = D["mle"]
    th_mle, th_red = thetas_from(mle, fixed), thetas_from(red, fixed)

    rows = []
    S_mle = np.empty((n_p, n_sims, T), dtype=np.uint8)
    S_red = np.empty((n_p, n_sims, T), dtype=np.uint8)
    for i in range(n_p):
        row, S_mle[i], S_red[i] = compare("mle", mle[i], raw[i], red[i], th_mle[i], th_red[i], plans[i],
                                          D["choices"][i], n_sims, seed, i * n_sims, ths)
        rows.append({"user_id": float(users[i]), "n_respuestas_rellenadas": int(D["n_imputed"][i]),
                     "n_ensayos_agregados": int(D["n_added"][i]), **row})

    out_csv = out_dir / "soluciones_reales.csv"
    write_csv(out_csv, rows)
    out_npz = out_dir / "simulaciones_reales.npz"
    np.savez_compressed(out_npz, user_id=users, choices=D["choices"], correct=D["correct"],
                        S_mle=S_mle, S_red=S_red, seed=np.int64(seed))
    for f in (out_csv, out_npz):
        os.chmod(f, 0o600)
    log(f"Parte A guardada en {out_csv} y {out_npz.name}")
    summarize(rows, "mle", ths, "Resumen parte A (participantes reales, techo = MLE)")


# ---------------------------------------------------------------- parte B
def part_sim(cfg, ann, sim, model_dir, net, fixed):
    grid = Grid(sim["grid"])
    R = int(sim["replicates"])
    N = grid.n_total * R
    seq = RandomSequences(sim)
    n = int(cfg["simulated"]["n_agents"])
    n_sims, ths = int(cfg["n_sims"]), cfg["thresholds"]
    seed = int(cfg["seed"]) + 1
    sp = ann["split"]

    rng = np.random.default_rng(int(cfg["seed"]))
    pool = np.empty(0, dtype=np.int64)
    while True:
        cand = rng.integers(0, N, size=4 * n, dtype=np.int64)
        cand = cand[split_of_points(cand // R, ann["seed"], sp["train"], sp["val"]) == 2]
        pool = np.concatenate([pool, cand])
        _, first = np.unique(pool, return_index=True)
        if first.size >= n:
            agents = np.sort(pool[np.sort(first)][:n])
            break
    gidx = agents // R
    th_true = grid.thetas(gidx)
    log(f"Parte B: {n} agentes de prueba al azar (de {N:,}), {n_sims} simulaciones por conjunto")

    ch, X = seq.simulate_features(th_true, agents.astype(np.uint64))   # mismas decisiones que el dataset
    raw = net.predict(X)
    red = clip_params(raw, cfg["clip"])
    th_red = thetas_from(red, fixed)
    true5 = th_true[:, :5]
    rs = []
    for j, p in enumerate(PARAMS):
        rs.append(f"{p} r={np.corrcoef(true5[:, j], raw[:, j])[0, 1]:.3f}")
    log("Control de la red exportada (deben parecerse a las metricas de prueba del entrenamiento): "
        + ", ".join(rs))

    rows = []
    t0 = time.time()
    for i in range(n):
        plan = seq.agent_plan(int(agents[i]))
        row, _, _ = compare("verdad", true5[i], raw[i], red[i], th_true[i], th_red[i], plan, ch[i],
                            n_sims, seed, i * n_sims, ths)
        rows.append({"agente": int(agents[i]), "punto_rejilla": int(gidx[i]), "replica": int(agents[i] % R),
                     **row})
        if (i + 1) % 1000 == 0 or i + 1 == n:
            log(f"  {i + 1}/{n} agentes ({time.time() - t0:.0f} s)")

    out_dir = model_dir / cfg["simulated"]["out_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "soluciones_simuladas.csv"
    write_csv(out_csv, rows)
    log(f"Parte B guardada en {out_csv}")
    summarize(rows, "verdad", ths, "Resumen parte B (agentes simulados, techo = parametros verdaderos)")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--parte", choices=("ambas", "reales", "simulados"), default="ambas")
    args = ap.parse_args(argv)

    cfg = json.load(open(args.config, encoding="utf-8"))
    ann = json.load(open(cfg["ann_config"], encoding="utf-8"))
    sim = json.load(open(ann["sim_config"], encoding="utf-8"))
    model_dir = _expand(sim["run_dir"]) / "ann" / ann["ann_run_name"]
    net = NumpyNet(model_dir / "modelo_numpy.npz")
    if net.targets != PARAMS:
        sys.exit(f"La red predice {net.targets}, se esperaba {PARAMS}.")
    fixed = Grid(sim["grid"]).fixed
    log(f"Red {model_dir.name}: {len(net.sel)} entradas | fijos {fixed} | semilla {cfg['seed']}")

    if args.parte in ("ambas", "reales"):
        part_real(cfg, net, fixed)
    if args.parte in ("ambas", "simulados"):
        part_sim(cfg, ann, sim, model_dir, net, fixed)


if __name__ == "__main__":
    main()
