"""Rejilla de parametros.

Cada eje guarda valores ENTEROS (por ejemplo centesimas: 1..99) y una escala
(100). Asi se evita el error de redondeo de los decimales en la computadora
(0.1 + 0.2 != 0.3) y cada valor tiene exactamente 2 decimales.

A cada combinacion de la rejilla le corresponde un indice global unico
(0 .. n_total-1), como una numeracion de casillas. Del indice se recuperan los
parametros, asi que no hace falta guardar los parametros de cada agente.
El primer eje de la lista cambia mas lento y el ultimo mas rapido.

Replicas: cada punto de la rejilla se simula R veces con azar distinto.
El agente numero i corresponde al punto i // R y a la replica i % R, de modo
que las R replicas de un mismo punto quedan juntas.
"""
import numpy as np
from .model import N_PARAMS, PARAM_NAMES

_DEFAULT_FIXED = {"beta": 17.0, "bias_wm": 0.0, "bias_rl": 1.0}


class Grid:
    def __init__(self, spec):
        self.axes = []
        for ax in spec["axes"]:
            name = ax["name"]
            if name not in PARAM_NAMES:
                raise ValueError(f"parametro desconocido en la rejilla: {name}")
            if "values" in ax:
                ints = np.array(ax["values"], dtype=np.int64)
            else:
                ints = np.arange(ax["start"], ax["stop"] + 1, ax["step"], dtype=np.int64)
            scale = float(ax.get("scale", 1))
            self.axes.append({"name": name, "ints": ints, "scale": scale})
        self.fixed = dict(_DEFAULT_FIXED, **spec.get("fixed", {}))
        names = [a["name"] for a in self.axes]
        missing = [p for p in PARAM_NAMES if p not in names and p not in self.fixed]
        if missing:
            raise ValueError(f"faltan parametros (ni en la rejilla ni fijos): {missing}")
        self.shape = tuple(len(a["ints"]) for a in self.axes)
        self.n_total = int(np.prod(self.shape, dtype=np.int64))

    @property
    def names(self):
        return [a["name"] for a in self.axes]

    def codes(self, idx):
        """Valores enteros de cada eje para los indices dados: (n, n_ejes)."""
        pos = np.unravel_index(np.asarray(idx, dtype=np.int64), self.shape)
        return np.stack([a["ints"][p] for a, p in zip(self.axes, pos)], axis=1)

    def thetas(self, idx):
        """Vectores de parametros (n, 8) listos para el modelo."""
        idx = np.asarray(idx, dtype=np.int64)
        th = np.empty((idx.size, N_PARAMS), dtype=np.float64)
        for k, v in self.fixed.items():
            th[:, PARAM_NAMES.index(k)] = v
        c = self.codes(idx)
        for j, a in enumerate(self.axes):
            th[:, PARAM_NAMES.index(a["name"])] = c[:, j] / a["scale"]
        return th

    def index_of(self, **values):
        """Indice global de una combinacion, p. ej. index_of(alpha=0.25, ..., K=3)."""
        pos = []
        for a in self.axes:
            target = round(values[a["name"]] * a["scale"])
            hit = np.flatnonzero(a["ints"] == target)
            if hit.size == 0:
                raise ValueError(f"{a['name']}={values[a['name']]} no esta en la rejilla")
            pos.append(int(hit[0]))
        return int(np.ravel_multi_index(pos, self.shape))

    def shard_bounds(self, shard_id, n_shards, replicates=1):
        """Rango [inicio, fin) de indices de AGENTE del fragmento shard_id."""
        return split_range(self.n_total * replicates, shard_id, n_shards)

    def code_dtype(self):
        """uint8 si todos los valores enteros caben en 0..255 (caso centesimas), si no uint16."""
        top = max(int(a["ints"].max()) for a in self.axes)
        return np.uint8 if top <= 255 else np.uint16


def split_range(n_total, shard_id, n_shards):
    """Divide 0..n_total-1 en n_shards rangos contiguos de tamano casi igual."""
    base, extra = divmod(n_total, n_shards)
    start = shard_id * base + min(shard_id, extra)
    end = start + base + (1 if shard_id < extra else 0)
    return start, end


def agent_to_grid(agent_idx, replicates):
    """Indice de agente -> (indice de rejilla, numero de replica).
    Las replicas de un mismo punto de la rejilla quedan juntas y seguidas."""
    agent_idx = np.asarray(agent_idx, dtype=np.int64)
    return agent_idx // replicates, agent_idx % replicates
