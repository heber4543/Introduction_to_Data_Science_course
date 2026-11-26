"""Entorno de la tarea RLWM: diseno -> plan.

- Diseno: descripcion general (tamanos de conjunto, presentaciones por estimulo,
  orden de bloques, regla de retrasos, regla de respuestas correctas).
- Plan: una realizacion concreta del diseno con una semilla. Es la secuencia
  exacta de estimulos y respuestas correctas que veran los agentes.

El plan es independiente del agente: se genera una vez, se guarda y cualquier
agente lo puede recorrer. Hoy todos los agentes usan el mismo plan; para que
cada agente tenga su propio plan basta generar varios planes y asignarlos.

Reglas portadas de:
- generar_entorno_fase (parameter_recovery.ipynb, tesis del usuario): fases,
  retraso entre presentaciones de 1 a 2*TC, regla de respuestas.
- lib/engine/sequence.ts y mapping.ts (app RLWM nueva): metodo uniform_delay
  (muchas secuencias candidatas, se elige la de distribucion de retrasos mas
  cercana a la uniforme, medida con distancia de variacion total) y regla
  distinct_then_random.
"""
import json
import hashlib
import numpy as np

DESIGNS = {
    # Fase 3 de la tesis: un bloque por tamano de conjunto (2, 3, 4), 10 presentaciones.
    "tesis_fase3": {"set_sizes": [2, 3, 4], "presentations": 10},
    "tesis_fase2": {"set_sizes": [2, 2, 2, 3, 3, 3, 4, 4, 4], "presentations": 10},
    "tesis_fase1": {"set_sizes": [2, 2, 2, 3, 3, 3, 4, 4, 5, 5], "presentations": 13},
}


def assign_actions(rng, ns, n_actions=3, method="distinct_then_random"):
    """Respuesta correcta de cada estimulo del bloque (0..n_actions-1)."""
    if method == "independent":
        return rng.integers(0, n_actions, ns)
    if method != "distinct_then_random":
        raise ValueError(f"metodo de asignacion desconocido: {method}")
    if ns <= n_actions:
        return rng.choice(n_actions, ns, replace=False)
    acts = np.concatenate([np.arange(n_actions), rng.integers(0, n_actions, ns - n_actions)])
    rng.shuffle(acts)
    return acts


def _attempt(rng, ns, presentations, max_delay):
    remaining = np.full(ns, presentations)
    last = np.full(ns, -1)
    T = ns * presentations
    seq = np.empty(T, dtype=np.int64)
    for t in range(T):
        cand = np.flatnonzero(remaining > 0)
        urgent = [i for i in cand if last[i] >= 0 and t - last[i] >= max_delay]
        if len(urgent) > 1:
            return None  # callejon sin salida: dos estimulos exceden el retraso maximo
        s = urgent[0] if urgent else cand[rng.integers(0, len(cand))]
        seq[t] = s
        remaining[s] -= 1
        last[s] = t
    return seq


def delays_of(seq):
    """Retraso de cada ensayo = ensayos desde la presentacion anterior del mismo
    estimulo (1 = repeticion inmediata); -1 en la primera presentacion."""
    last = {}
    out = np.empty(len(seq), dtype=np.int64)
    for t, s in enumerate(seq):
        out[t] = t - last[s] if s in last else -1
        last[s] = t
    return out


def _tv_from_uniform(seq, max_delay):
    d = delays_of(seq)
    d = d[d > 0]
    hist = np.bincount(d, minlength=max_delay + 1)[1:max_delay + 1].astype(float)
    hist /= hist.sum()
    return 0.5 * np.abs(hist - 1.0 / max_delay).sum()


def generate_sequence(rng, ns, presentations, max_delay_factor=2, n_candidates=300):
    """Secuencia de un bloque con retraso <= max_delay_factor*ns, la mas cercana a
    una distribucion uniforme de retrasos entre n_candidates secuencias validas."""
    max_delay = max_delay_factor * ns
    best, best_err, found, tries = None, np.inf, 0, 0
    while found < n_candidates and tries < 200 * n_candidates:
        tries += 1
        seq = _attempt(rng, ns, presentations, max_delay)
        if seq is None:
            continue
        found += 1
        err = _tv_from_uniform(seq, max_delay)
        if err < best_err:
            best, best_err = seq, err
    if best is None:
        raise RuntimeError(f"no se encontro secuencia valida para TC={ns}")
    return best, float(best_err)


def make_plan(design, seed):
    """Genera un plan a partir de un diseno (dict) y una semilla entera."""
    if isinstance(design, str):
        name = design
        design = dict(DESIGNS[design], name=name)
    rng = np.random.default_rng(seed)
    n_actions = int(design.get("n_actions", 3))
    set_sizes = list(design["set_sizes"])
    pres = design["presentations"]
    pres_list = list(pres) if isinstance(pres, (list, tuple)) else [pres] * len(set_sizes)
    order = np.arange(len(set_sizes))
    if design.get("shuffle_blocks", True):
        rng.shuffle(order)
    mapping_method = design.get("mapping", "distinct_then_random")
    factor = design.get("max_delay_factor", 2)
    n_cand = design.get("n_candidates", 300)

    cols = {k: [] for k in ("new_block", "block", "stim", "ns", "correct", "iteration", "delay")}
    blocks_meta = []
    for b, idx in enumerate(order):
        ns, P = int(set_sizes[idx]), int(pres_list[idx])
        mapping = assign_actions(rng, ns, n_actions, mapping_method)
        seq, tv = generate_sequence(rng, ns, P, factor, n_cand)
        d = delays_of(seq)
        it = np.zeros(ns, dtype=int)
        for t, s in enumerate(seq):
            it[s] += 1
            cols["new_block"].append(t == 0)
            cols["block"].append(b)
            cols["stim"].append(s)
            cols["ns"].append(ns)
            cols["correct"].append(mapping[s])
            cols["iteration"].append(it[s])
            cols["delay"].append(d[t])
        blocks_meta.append({"block": b, "set_size": ns, "presentations": P,
                            "mapping": [int(x) for x in mapping],
                            "delay_tv_from_uniform": round(tv, 4)})

    plan = {
        "new_block": np.array(cols["new_block"], dtype=np.bool_),
        "block": np.array(cols["block"], dtype=np.int16),
        "stim": np.array(cols["stim"], dtype=np.int8),
        "ns": np.array(cols["ns"], dtype=np.int8),
        "correct": np.array(cols["correct"], dtype=np.int8),
        "iteration": np.array(cols["iteration"], dtype=np.int16),
        "delay": np.array(cols["delay"], dtype=np.int16),
        "n_actions": n_actions,
        "max_ns": int(max(set_sizes)),
    }
    plan["meta"] = {"design": design, "seed": int(seed), "n_trials": int(len(plan["stim"])),
                    "blocks": blocks_meta, "plan_hash": plan_hash(plan)}
    return plan


_ARRAY_KEYS = ("new_block", "block", "stim", "ns", "correct", "iteration", "delay")


def plan_hash(plan):
    h = hashlib.sha256()
    for k in _ARRAY_KEYS:
        h.update(np.ascontiguousarray(plan[k]).tobytes())
    h.update(str((plan["n_actions"], plan["max_ns"])).encode())
    return h.hexdigest()[:16]


def save_plan(plan, path_npz, path_json):
    np.savez(path_npz, **{k: plan[k] for k in _ARRAY_KEYS},
             n_actions=plan["n_actions"], max_ns=plan["max_ns"])
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(plan["meta"], f, indent=2, ensure_ascii=False)


def load_plan(path_npz):
    z = np.load(path_npz)
    plan = {k: z[k] for k in _ARRAY_KEYS}
    plan["n_actions"] = int(z["n_actions"]); plan["max_ns"] = int(z["max_ns"])
    return plan
