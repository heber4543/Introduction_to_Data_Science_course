"""Secuencia aleatoria por agente.

Cada agente recibe su propia secuencia de la tarea, generada a partir de
(plan_seed, numero de agente). No se guarda: el mismo codigo con la misma
semilla regenera siempre la misma secuencia para el mismo agente.

Regla (como la app de la tesis, rlwm-main, y generar_entorno_fase):
- orden de bloques al azar;
- respuestas correctas distinct_then_random (TC <= 3: botones distintos;
  TC > 3: cada boton al menos una vez y el resto al azar, todo barajado);
- secuencia al azar con retraso entre presentaciones del mismo estimulo de
  1 a max_delay_factor * TC (sin optimizar la uniformidad del retraso).

Ademas incluye el nucleo que, para cada agente, genera su secuencia, simula
sus elecciones y calcula sus 59 metricas en un solo paso.
"""
import numpy as np
from numba import njit, prange

from .environment import DESIGNS
from .features import _features_one, NF, GLOBAL, feature_names
from .model import _splitmix64, _simulate_one, _U64_GOLDEN, _SH11, _INV_2_53

MAPPING_MODES = {"distinct_then_random": 0, "independent": 1}


@njit(cache=True, inline="always")
def _draw(base, ctr, n):
    """Entero al azar en [0, n) del flujo del agente; avanza el contador."""
    h = _splitmix64(base + np.uint64(ctr[0]) * _U64_GOLDEN)
    ctr[0] += 1
    v = int(np.float64(h >> _SH11) * _INV_2_53 * n)
    return v if v < n else n - 1


@njit(cache=True, inline="always")
def _shuffle(a, n, base, ctr):
    for i in range(n - 1, 0, -1):
        j = _draw(base, ctr, i + 1)
        tmp = a[i]; a[i] = a[j]; a[j] = tmp


@njit(cache=True)
def _gen_plan(agent_id, plan_seed, set_sizes, pres_blocks, factor, shuffle_blocks,
              mapping_mode, nA, tcs, new_block, block, stim, ns_arr, correct,
              iteration, tc_idx, pres):
    base = _splitmix64(plan_seed ^ _splitmix64(agent_id * _U64_GOLDEN))
    ctr = np.zeros(1, np.int64)
    nb = set_sizes.shape[0]
    order = np.arange(nb)
    if shuffle_blocks:
        _shuffle(order, nb, base, ctr)
    max_ns = 0; max_len = 0
    for b in range(nb):
        if set_sizes[b] > max_ns:
            max_ns = set_sizes[b]
        if set_sizes[b] * pres_blocks[b] > max_len:
            max_len = set_sizes[b] * pres_blocks[b]
    remaining = np.empty(max_ns, np.int64)
    last = np.empty(max_ns, np.int64)
    cand = np.empty(max_ns, np.int64)
    its = np.empty(max_ns, np.int64)
    mapping = np.empty(max_ns, np.int64)
    perm = np.empty(nA, np.int64)
    seq = np.empty(max_len, np.int64)
    t0 = 0
    for b in range(nb):
        ns = set_sizes[order[b]]
        P = pres_blocks[order[b]]
        # respuestas correctas
        if mapping_mode == 0:
            for a in range(nA):
                perm[a] = a
            _shuffle(perm, nA, base, ctr)
            if ns <= nA:
                for i in range(ns):
                    mapping[i] = perm[i]
            else:
                for i in range(nA):
                    mapping[i] = perm[i]
                for i in range(nA, ns):
                    mapping[i] = _draw(base, ctr, nA)
                _shuffle(mapping, ns, base, ctr)
        else:
            for i in range(ns):
                mapping[i] = _draw(base, ctr, nA)
        # secuencia con retraso maximo factor*TC
        T_b = ns * P
        maxd = factor * ns
        ok = False
        for attempt in range(10000):
            for i in range(ns):
                remaining[i] = P; last[i] = -1
            ok = True
            for t in range(T_b):
                nc = 0; urgent = -1; nurg = 0
                for i in range(ns):
                    if remaining[i] > 0:
                        cand[nc] = i; nc += 1
                        if last[i] >= 0 and t - last[i] >= maxd:
                            urgent = i; nurg += 1
                if nurg > 1:
                    ok = False
                    break
                s = urgent if nurg == 1 else cand[_draw(base, ctr, nc)]
                seq[t] = s; remaining[s] -= 1; last[s] = t
            if ok:
                break
        if not ok:
            raise ValueError("no se encontro una secuencia valida para un bloque")
        for i in range(ns):
            its[i] = 0
        k = 0
        while tcs[k] != ns:
            k += 1
        for t in range(T_b):
            s = seq[t]
            its[s] += 1
            g = t0 + t
            new_block[g] = t == 0
            block[g] = b
            stim[g] = s
            ns_arr[g] = ns
            correct[g] = mapping[s]
            iteration[g] = its[s]
            tc_idx[g] = k
            pres[g] = P
        t0 += T_b


@njit(parallel=True, cache=True)
def sim_features_batch(thetas, agent_ids, sim_seed, plan_seed, set_sizes, pres_blocks, factor,
                       shuffle_blocks, mapping_mode, nA, max_ns, T, tcs, out_choices, out_feat):
    """Para cada agente: genera su secuencia, simula sus elecciones y calcula sus metricas."""
    n_tc = tcs.shape[0]
    for i in prange(thetas.shape[0]):
        new_block = np.empty(T, np.bool_)
        block = np.empty(T, np.int64); stim = np.empty(T, np.int64)
        ns_arr = np.empty(T, np.int64); correct = np.empty(T, np.int64)
        iteration = np.empty(T, np.int64); tc_idx = np.empty(T, np.int64)
        pres = np.empty(T, np.int64)
        _gen_plan(agent_ids[i], plan_seed, set_sizes, pres_blocks, factor, shuffle_blocks,
                  mapping_mode, nA, tcs, new_block, block, stim, ns_arr, correct,
                  iteration, tc_idx, pres)
        _simulate_one(thetas[i], agent_ids[i], sim_seed, new_block, stim, ns_arr, correct,
                      nA, max_ns, out_choices[i])
        _features_one(out_choices[i], new_block, stim, ns_arr, correct, iteration,
                      tc_idx, pres, n_tc, max_ns, out_feat[i])


class RandomSequences:
    """Configuracion de las secuencias aleatorias por agente, leida de la corrida."""

    def __init__(self, cfg):
        design = cfg["design"]
        d = dict(DESIGNS[design]) if isinstance(design, str) else dict(design)
        sq = cfg.get("sequence", {})
        self.set_sizes = np.array(d["set_sizes"], dtype=np.int64)
        p = d["presentations"]
        self.pres_blocks = (np.array(p, dtype=np.int64) if isinstance(p, (list, tuple))
                            else np.full(len(self.set_sizes), int(p), dtype=np.int64))
        self.factor = int(sq.get("max_delay_factor", 2))
        self.shuffle_blocks = bool(sq.get("shuffle_blocks", True))
        self.mapping_mode = MAPPING_MODES[sq.get("mapping", "distinct_then_random")]
        self.nA = int(d.get("n_actions", 3))
        self.max_ns = int(self.set_sizes.max())
        self.T = int((self.set_sizes * self.pres_blocks).sum())
        self.tcs = np.unique(self.set_sizes).astype(np.int64)
        self.plan_seed = np.uint64(cfg["plan_seed"])
        self.sim_seed = np.uint64(cfg["sim_seed"])
        self.n_features = len(self.tcs) * NF + len(GLOBAL)

    def names(self):
        """Nombres de las metricas, en el mismo orden que las columnas calculadas."""
        return feature_names({"ns": self.tcs})

    def simulate_features(self, thetas, agent_ids):
        """(n, 8) parametros y (n,) indices de agente -> elecciones (n, T) uint8 y metricas (n, F) float32."""
        thetas = np.ascontiguousarray(np.atleast_2d(thetas), dtype=np.float64)
        agent_ids = np.ascontiguousarray(agent_ids, dtype=np.uint64)
        ch = np.empty((thetas.shape[0], self.T), dtype=np.uint8)
        X = np.empty((thetas.shape[0], self.n_features), dtype=np.float32)
        sim_features_batch(thetas, agent_ids, self.sim_seed, self.plan_seed, self.set_sizes,
                           self.pres_blocks, self.factor, self.shuffle_blocks, self.mapping_mode,
                           self.nA, self.max_ns, self.T, self.tcs, ch, X)
        return ch, X

    def agent_plan(self, agent_id):
        """Plan (secuencia) de un agente, en el mismo formato que environment.make_plan."""
        T = self.T
        a = {k: np.empty(T, np.int64) for k in ("block", "stim", "ns", "correct", "iteration", "tc_idx", "pres")}
        nb = np.empty(T, np.bool_)
        _gen_plan(np.uint64(agent_id), self.plan_seed, self.set_sizes, self.pres_blocks, self.factor,
                  self.shuffle_blocks, self.mapping_mode, self.nA, self.tcs, nb, a["block"], a["stim"],
                  a["ns"], a["correct"], a["iteration"], a["tc_idx"], a["pres"])
        return {"new_block": nb, "block": a["block"].astype(np.int16), "stim": a["stim"].astype(np.int8),
                "ns": a["ns"].astype(np.int8), "correct": a["correct"].astype(np.int8),
                "iteration": a["iteration"].astype(np.int16), "n_actions": self.nA, "max_ns": self.max_ns}
