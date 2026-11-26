"""Paso 8. Entrena una ANN densa (MLP) que predice alpha, phi, rho, epsilon y K
a partir de las 59 metricas de desempeno, con seleccion de caracteristicas
embebida, y la evalua en un conjunto de prueba sintetico.

Corre con el Python base del cluster (module load python/miniforge), que trae
PyTorch. Usa GPU si la hay y la configuracion lo permite; si no, CPU.

Uso:  python -m rlwm_sim.train_ann configs/ann_v2.json            (corrida completa)
      python -m rlwm_sim.train_ann configs/ann_v2.json --smoke    (prueba corta con 2 archivos)

Procedimiento
1. Carga en RAM las metricas (float16) y los parametros de todos los agentes.
2. Divide por punto de la rejilla en entrenamiento, validacion y prueba (60/20/20
   por defecto) con una funcion hash del indice del punto: las replicas de un
   punto quedan siempre del mismo lado y la division es reproducible.
3. Imputa NaN con la mediana y estandariza, ambas calculadas con entrenamiento.
4. Seleccion embebida: para cada lambda, entrena una fraccion de pasada con
   lasso por grupos en la primera capa aplicado con su operador proximal
   (Scardapane et al., 2017), que lleva a cero exacto las columnas de pesos de
   las entradas que no aportan. Se elige la lambda con menos caracteristicas cuya
   perdida de validacion quede dentro de la tolerancia de la mejor.
5. Entrenamiento final con las seleccionadas, sin penalizacion, por pasadas
   completas (cada agente de entrenamiento una vez por pasada, en orden aleatorio),
   con paro temprano por validacion; se conserva la mejor red.
6. Metricas en validacion y prueba (parametros verdaderos). La red se guarda
   para evaluarla despues con participantes reales.
K se predice como numero y se redondea a 2, 3 o 4.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

TARGETS = ("alpha", "phi", "rho", "epsilon", "K")
T_OFFSET = np.array([0, 0, 0, 0, 2], dtype=np.float32)   # objetivo escalado = (valor - offset) / escala
T_SCALE = np.array([1, 1, 1, 0.5, 2], dtype=np.float32)   # asi los cinco quedan entre 0 y 1


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- division por punto de la rejilla
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


def split_indices(N, R, seed, f_train, f_val, chunk=20_000_000):
    parts = ([], [], [])
    for a in range(0, N, chunk):
        idx = np.arange(a, min(a + chunk, N), dtype=np.int64)
        s = split_of_points(idx // R, seed, f_train, f_val)
        for k in range(3):
            parts[k].append(idx[s == k])
    return tuple(np.concatenate(p) for p in parts)


# ---------------------------------------------------------------- datos
def load_features(fdir, n_shards, max_files=None):
    files = [fdir / f"features_{s:05d}.npz" for s in range(n_shards)]
    if max_files:
        files = files[:max_files]
    bounds = []
    for f in files:
        with np.load(f) as z:
            bounds.append((int(z["agent_start"]), int(z["agent_end"])))
            R = int(z["replicates"]); ncod = z["codes"].shape[1]
    if bounds[0][0] != 0 or any(bounds[i][1] != bounds[i + 1][0] for i in range(len(bounds) - 1)):
        sys.exit("Los archivos de caracteristicas no son contiguos desde el agente 0.")
    N = bounds[-1][1]
    names = json.load(open(fdir / "feature_names.json"))["names"]
    log(f"Reservando memoria: {N:,} agentes x {len(names)} caracteristicas "
        f"({N * len(names) * 2 / 1e9:.1f} GB en float16)")
    X = np.empty((N, len(names)), dtype=np.float16)
    C = np.empty((N, ncod), dtype=np.uint8)
    t0 = time.time()
    for i, (f, (a, b)) in enumerate(zip(files, bounds)):
        with np.load(f) as z:
            X[a:b] = z["X"]
            C[a:b] = z["codes"]
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            log(f"  cargados {i + 1}/{len(files)} archivos ({time.time() - t0:.0f} s)")
    return X, C, R, names


def target_matrix(codes, axes):
    """Codigos enteros de la rejilla -> objetivos escalados (n, 5) float32."""
    y = np.empty((codes.shape[0], len(TARGETS)), dtype=np.float32)
    axis_names = [a["name"] for a in axes]
    for j, t in enumerate(TARGETS):
        k = axis_names.index(t)
        y[:, j] = codes[:, k].astype(np.float32) / float(axes[k].get("scale", 1))
    return (y - T_OFFSET) / T_SCALE


def to_original(y_scaled):
    return y_scaled * T_SCALE + T_OFFSET


# ---------------------------------------------------------------- modelo
class MLP(nn.Module):
    def __init__(self, n_in, hidden, n_out=len(TARGETS)):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    @property
    def first(self):
        return self.net[0]

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def prox_group_lasso(W, thresh):
    """Operador proximal del lasso por grupos: encoge cada columna (una entrada)
    y la vuelve exactamente cero si su norma es menor que thresh."""
    norms = W.norm(dim=0)
    W.mul_(torch.clamp(1 - thresh / (norms + 1e-12), min=0))


class Prep:
    """Imputacion por mediana y estandarizacion."""
    def __init__(self, med, mean, std, dev):
        self.med = torch.tensor(med, device=dev)
        self.mean = torch.tensor(mean, device=dev)
        self.std = torch.tensor(std, device=dev)

    def __call__(self, xb):
        x = xb.float()
        x = torch.where(torch.isnan(x), self.med, x)
        return (x - self.mean) / self.std


# ---------------------------------------------------------------- evaluacion
def _metrics(pred, true):
    """pred y true en unidades originales (n, 5)."""
    out = {}
    for j, t in enumerate(TARGETS):
        p, y = pred[:, j].astype(np.float64), true[:, j].astype(np.float64)
        err = p - y
        ss_res = float((err ** 2).sum()); ss_tot = float(((y - y.mean()) ** 2).sum())
        out[t] = {"pearson_r": float(np.corrcoef(p, y)[0, 1]) if p.std() > 0 and y.std() > 0 else float("nan"),
                  "rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean()),
                  "r2": 1 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "n": int(y.size)}
    k = TARGETS.index("K")
    kr = np.clip(np.rint(pred[:, k]), 2, 4)
    out["K"]["accuracy_rounded"] = float((kr == true[:, k]).mean())
    labels = [2, 3, 4]
    out["K"]["confusion_true_rows_pred_cols"] = [[int(((true[:, k] == a) & (kr == b)).sum()) for b in labels]
                                                 for a in labels]
    return out


@torch.no_grad()
def predict_rows(model, X, idx, prep, sel_t, dev, chunk=1_000_000):
    model.eval()
    outs = []
    for a in range(0, len(idx), chunk):
        r = idx[a:a + chunk]
        xb = prep(torch.from_numpy(X[r]).to(dev))[:, sel_t]
        outs.append(model(xb).float().cpu().numpy())
    model.train()
    return np.concatenate(outs)


def _log_metrics(title, m):
    log(title)
    for t in TARGETS:
        extra = f" | exactitud K redondeada {m[t]['accuracy_rounded']:.3f}" if t == "K" else ""
        log(f"  {t:8s} r={m[t]['pearson_r']:.3f} RMSE={m[t]['rmse']:.4f} "
            f"MAE={m[t]['mae']:.4f} R2={m[t]['r2']:.3f}{extra}")


# ---------------------------------------------------------------- entrenamiento
class EpochBatches:
    """Recorre el entrenamiento por pasadas: cada pasada usa a cada agente de
    entrenamiento una vez, en orden aleatorio."""
    def __init__(self, train_idx, B, rng):
        self.idx, self.B, self.rng = train_idx, B, rng
        self.steps_per_pass = math.ceil(len(train_idx) / B)
        self.perm, self.k = None, 0

    def next(self):
        if self.perm is None or self.k >= self.steps_per_pass:
            self.perm = self.rng.permutation(len(self.idx))
            self.k = 0
        b = self.perm[self.k * self.B:(self.k + 1) * self.B]
        self.k += 1
        return np.sort(self.idx[b])


def train(cfg, X, C, axes, train_idx, mon_idx, prep, sel, lam, steps, eval_every, patience,
          dev, tag, log_rows, seed):
    torch.manual_seed(seed)
    sel_t = torch.tensor(sel, device=dev)
    model = MLP(len(sel), cfg["hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=cfg["lr"] * 0.01)
    batches = EpochBatches(train_idx, cfg["batch_size"], np.random.default_rng(seed))
    y_mon = target_matrix(C[mon_idx], axes)
    best, best_state, bad = float("inf"), None, 0
    pool = ThreadPoolExecutor(max_workers=1)

    def fetch():
        idx = batches.next()
        return X[idx], C[idx]

    fut = pool.submit(fetch)
    t0 = time.time()
    run_loss, n_run = 0.0, 0
    for step in range(1, steps + 1):
        xb_np, cb_np = fut.result()
        fut = pool.submit(fetch)
        xb = prep(torch.from_numpy(xb_np).to(dev))[:, sel_t]
        yb = torch.from_numpy(target_matrix(cb_np, axes)).to(dev)
        loss = ((model(xb) - yb) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if lam > 0:
            prox_group_lasso(model.first.weight, sched.get_last_lr()[0] * lam)
        sched.step()
        run_loss += float(loss); n_run += 1
        if step == 100:
            rate = 100 / (time.time() - t0)
            log(f"[{tag}] {rate:.1f} pasos/s | una pasada ~{batches.steps_per_pass / rate / 60:.1f} min "
                f"| esta etapa como maximo ~{steps / rate / 60:.0f} min")
        if step % eval_every == 0 or step == steps:
            pm = predict_rows(model, X, mon_idx, prep, sel_t, dev)
            vl = float(((pm - y_mon) ** 2).mean())
            nz = int((model.first.weight.norm(dim=0) > 0).sum())
            pasada = step / batches.steps_per_pass
            log(f"[{tag}] paso {step}/{steps} (pasada {pasada:.2f}) perdida_ent {run_loss / n_run:.5f} "
                f"perdida_val {vl:.5f} entradas_activas {nz}/{len(sel)}")
            log_rows.append({"etapa": tag, "lambda": lam, "paso": step, "pasada": round(pasada, 3),
                             "perdida_entrenamiento": run_loss / n_run,
                             "perdida_validacion": vl, "entradas_activas": nz,
                             "segundos": round(time.time() - t0, 1)})
            run_loss, n_run = 0.0, 0
            if vl < best * (1 - 1e-4):
                best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if patience and bad >= patience:
                    log(f"[{tag}] paro temprano en el paso {step} (pasada {pasada:.2f})")
                    break
    pool.shutdown(wait=True)
    model.load_state_dict(best_state)
    return model, best, batches.steps_per_pass


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--smoke", action="store_true", help="prueba corta: 2 archivos y pocos pasos")
    args = ap.parse_args(argv)

    cfg = json.load(open(args.config, encoding="utf-8"))
    sim_cfg = json.load(open(cfg["sim_config"], encoding="utf-8"))
    run_dir = Path(os.path.expandvars(sim_cfg["run_dir"]))
    man = json.load(open(run_dir / "manifest.json"))
    axes = sim_cfg["grid"]["axes"]
    name = cfg["ann_run_name"] + ("_smoke" if args.smoke else "")
    out_dir = run_dir / "ann" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    torch.set_num_threads(threads)
    want = cfg.get("device", "auto")
    dev = torch.device("cuda" if (want in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
    log(f"Dispositivo: {dev} | hilos CPU {threads} | torch {torch.__version__}")

    sel_cfg, fin_cfg = dict(cfg["select"]), dict(cfg["final"])
    sp = cfg["split"]
    X, C, R, names = load_features(run_dir / "features", man["n_shards"], 2 if args.smoke else None)
    N = X.shape[0]
    train_idx, val_idx, test_idx = split_indices(N, R, cfg["seed"], sp["train"], sp["val"])
    log(f"Division por punto de la rejilla: entrenamiento {len(train_idx):,}, "
        f"validacion {len(val_idx):,}, prueba {len(test_idx):,} agentes")
    rng = np.random.default_rng(cfg["seed"])
    mon_rows = 20_000 if args.smoke else cfg["val_monitor_rows"]
    mon_idx = np.sort(rng.choice(val_idx, size=min(mon_rows, len(val_idx)), replace=False))

    stats_rows = 100_000 if args.smoke else cfg["stats_rows"]
    s_idx = np.sort(rng.choice(train_idx, size=min(stats_rows, len(train_idx)), replace=False))
    S = X[s_idx].astype(np.float32)
    med = np.nanmedian(S, axis=0)
    med = np.where(np.isnan(med), 0.0, med).astype(np.float32)
    S = np.where(np.isnan(S), med, S)
    mean = S.mean(axis=0).astype(np.float32)
    std = S.std(axis=0).astype(np.float32)
    std = np.where(std > 0, std, 1.0).astype(np.float32)
    del S
    prep = Prep(med, mean, std, dev)
    log_rows = []

    steps_per_pass = math.ceil(len(train_idx) / cfg["batch_size"])
    if args.smoke:
        sel_steps, sel_eval, lambdas = 200, 100, sel_cfg["lambdas"][:1] + sel_cfg["lambdas"][-1:]
        fin_steps, fin_eval, patience = 400, 100, 0
    else:
        sel_steps = max(1, int(sel_cfg["passes"] * steps_per_pass))
        sel_eval = max(1, steps_per_pass // sel_cfg["evals_per_pass"])
        lambdas = sel_cfg["lambdas"]
        fin_steps = int(fin_cfg["max_passes"] * steps_per_pass)
        fin_eval = max(1, steps_per_pass // fin_cfg["evals_per_pass"])
        patience = fin_cfg["patience_passes"] * fin_cfg["evals_per_pass"]
    log(f"Pasos por pasada: {steps_per_pass:,} | seleccion {len(lambdas)} x {sel_steps:,} pasos "
        f"| final hasta {fin_steps:,} pasos")

    # ---- 1. seleccion embebida
    all_sel = list(range(len(names)))
    results = []
    for lam in lambdas:
        model, vl, _ = train(cfg, X, C, axes, train_idx, mon_idx, prep, all_sel, lam, sel_steps,
                             sel_eval, 0, dev, f"seleccion_lambda_{lam}", log_rows, cfg["seed"])
        norms = model.first.weight.norm(dim=0).detach().cpu().numpy()
        results.append({"lambda": lam, "perdida_validacion": vl, "n_seleccionadas": int((norms > 0).sum()),
                        "norma_por_caracteristica": {n: float(v) for n, v in zip(names, norms)}})
    best_vl = min(r["perdida_validacion"] for r in results)
    ok = [r for r in results if r["perdida_validacion"] <= best_vl * (1 + sel_cfg["tolerance"])
          and r["n_seleccionadas"] > 0]
    chosen = min(ok, key=lambda r: (r["n_seleccionadas"], -r["lambda"]))
    sel = [i for i, n in enumerate(names) if chosen["norma_por_caracteristica"][n] > 0]
    log(f"Lambda elegida {chosen['lambda']}: {len(sel)} de {len(names)} caracteristicas")
    json.dump({"resultados": results, "lambda_elegida": chosen["lambda"],
               "tolerancia": sel_cfg["tolerance"], "seleccionadas": [names[i] for i in sel],
               "descartadas": [n for i, n in enumerate(names) if i not in sel]},
              open(out_dir / "seleccion.json", "w"), indent=2, ensure_ascii=False)
    _write_csv(out_dir / "registro_entrenamiento.csv", log_rows)

    # ---- 2. entrenamiento final
    model, vl, _ = train(cfg, X, C, axes, train_idx, mon_idx, prep, sel, 0.0, fin_steps, fin_eval,
                         patience, dev, "final", log_rows, cfg["seed"] + 7)
    _write_csv(out_dir / "registro_entrenamiento.csv", log_rows)
    torch.save({"state_dict": model.state_dict(), "hidden": cfg["hidden"], "selected": sel,
                "names": names, "median": med, "mean": mean, "std": std,
                "target_offset": T_OFFSET, "target_scale": T_SCALE, "targets": TARGETS,
                "split": sp, "seed": cfg["seed"], "config": cfg},
               out_dir / "modelo.pt")
    log(f"Red guardada en {out_dir / 'modelo.pt'}")

    # ---- 3. validacion y prueba
    sel_t = torch.tensor(sel, device=dev)
    for nombre, idx in (("validacion", val_idx), ("prueba", test_idx)):
        p = to_original(predict_rows(model, X, idx, prep, sel_t, dev))
        t = to_original(target_matrix(C[idx], axes))
        m = _metrics(p, t)
        json.dump(m, open(out_dir / f"metricas_{nombre}.json", "w"), indent=2)
        _log_metrics(f"Metricas en {nombre} ({len(idx):,} agentes, parametros verdaderos):", m)
        if nombre == "prueba":
            k = rng.choice(len(idx), size=min(200_000, len(idx)), replace=False)
            np.savez(out_dir / "muestra_predicciones_prueba.npz", agente=idx[k], verdadero=t[k],
                     predicho=p[k], objetivos=np.array(TARGETS))
    log(f"Resultados en {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
