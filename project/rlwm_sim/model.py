"""Nucleo del modelo RLWM (Reinforcement Learning + Working Memory).

Un solo conjunto de ecuaciones se usa para tres cosas, para que nunca
discrepen entre si:
  - simulate_batch: genera las elecciones de muchos agentes.
  - loglik_batch:   calcula la log-verosimilitud negativa (NLL) de elecciones dadas.
  - replay:         reconstruye ensayo a ensayo todo lo que "penso" un agente
                    (politicas, peso de WM, valores Q y WM) a partir de sus elecciones.

Ecuaciones (Collins y Frank, 2012; Collins et al., 2014; Master et al., 2020;
codigo RLWM.m de WMH-main de A. Collins; y el modelo de la tesis del usuario):
  w      = rho * min(1, K / ns)
  pi_RL  = softmax(beta * Q[s])
  pi_WM  = softmax(beta * WM[s])
  pi     = (1 - eps) * (w * pi_WM + (1 - w) * pi_RL) + eps / nA
  despues de elegir a y recibir r (1 acierto, 0 error):
  WM    <- WM + phi * (1/nA - WM)                (decaimiento de toda la WM)
  WM[s,a] <- WM[s,a] + aWM * (r - WM[s,a])       aWM = 1 si r=1, bias_wm si r=0
  Q[s,a]  <- Q[s,a]  + aRL * (r - Q[s,a])        aRL = alpha si r=1, alpha*bias_rl si r=0
Q y WM se reinician a 1/nA al inicio de cada bloque.

Preset "tesis": beta=17, bias_wm=0 (WM solo aprende de aciertos), bias_rl=1.
"""
import numpy as np
from numba import njit, prange

# Posiciones dentro del vector de parametros (una fila por agente)
P_ALPHA, P_PHI, P_RHO, P_EPS, P_K, P_BETA, P_BIAS_WM, P_BIAS_RL = range(8)
N_PARAMS = 8
PARAM_NAMES = ("alpha", "phi", "rho", "epsilon", "K", "beta", "bias_wm", "bias_rl")

PROB_FLOOR = 1e-10  # igual que el piso usado en el codigo de la tesis

# ---------------------------------------------------------------------------
# Generador aleatorio por contador (SplitMix64; Steele, Lea y Flood, 2014).
# El numero aleatorio del ensayo t del agente i depende solo de (semilla, i, t).
# Asi cada agente tiene su propia secuencia independiente, el calculo en paralelo
# da exactamente el mismo resultado sin importar como se repartan los agentes,
# y no hace falta guardar una semilla por agente.
# ---------------------------------------------------------------------------
_U64_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_U64_M1 = np.uint64(0xBF58476D1CE4E5B9)
_U64_M2 = np.uint64(0x94D049BB133111EB)
_SH30 = np.uint64(30)
_SH27 = np.uint64(27)
_SH31 = np.uint64(31)
_SH11 = np.uint64(11)
_INV_2_53 = 1.0 / 9007199254740992.0


@njit(cache=True, inline="always")
def _splitmix64(x):
    z = x + _U64_GOLDEN
    z = (z ^ (z >> _SH30)) * _U64_M1
    z = (z ^ (z >> _SH27)) * _U64_M2
    return z ^ (z >> _SH31)


@njit(cache=True, inline="always")
def uniform_for(seed, agent, trial):
    """Numero uniforme en [0, 1) para (semilla, agente, ensayo)."""
    h = _splitmix64(seed ^ _splitmix64(agent * _U64_GOLDEN + np.uint64(trial)))
    return np.float64(h >> _SH11) * _INV_2_53


# ---------------------------------------------------------------------------
# Piezas compartidas: politica y actualizacion
# ---------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _softmax_row(beta, row, nA, out):
    m = row[0]
    for a in range(1, nA):
        if row[a] > m:
            m = row[a]
    tot = 0.0
    for a in range(nA):
        out[a] = np.exp(beta * (row[a] - m))
        tot += out[a]
    for a in range(nA):
        out[a] /= tot


@njit(cache=True, inline="always")
def _policy(Q, WM, s, w, beta, eps, nA, p_rl, p_wm, p):
    _softmax_row(beta, Q[s], nA, p_rl)
    _softmax_row(beta, WM[s], nA, p_wm)
    for a in range(nA):
        p[a] = (1.0 - eps) * (w * p_wm[a] + (1.0 - w) * p_rl[a]) + eps / nA


@njit(cache=True, inline="always")
def _update(Q, WM, s, a, r, alpha, phi, bias_wm, bias_rl, ns, nA):
    q0 = 1.0 / nA
    for i in range(ns):
        for j in range(nA):
            WM[i, j] += phi * (q0 - WM[i, j])
    a_wm = 1.0 if r == 1 else bias_wm
    WM[s, a] += a_wm * (r - WM[s, a])
    a_rl = alpha if r == 1 else alpha * bias_rl
    Q[s, a] += a_rl * (r - Q[s, a])


@njit(cache=True, inline="always")
def _reset(Q, WM, ns, nA):
    q0 = 1.0 / nA
    for i in range(ns):
        for j in range(nA):
            Q[i, j] = q0
            WM[i, j] = q0


# ---------------------------------------------------------------------------
# Simulacion
# ---------------------------------------------------------------------------
@njit(cache=True)
def _simulate_one(theta, agent_id, seed, new_block, stim, ns_arr, correct,
                  nA, max_ns, out_row):
    alpha = theta[P_ALPHA]; phi = theta[P_PHI]; rho = theta[P_RHO]
    eps = theta[P_EPS]; K = theta[P_K]; beta = theta[P_BETA]
    bias_wm = theta[P_BIAS_WM]; bias_rl = theta[P_BIAS_RL]
    Q = np.empty((max_ns, nA)); WM = np.empty((max_ns, nA))
    p_rl = np.empty(nA); p_wm = np.empty(nA); p = np.empty(nA)
    w = 0.0
    T = stim.shape[0]
    for t in range(T):
        ns = ns_arr[t]
        if new_block[t]:
            _reset(Q, WM, ns, nA)
            w = rho * min(1.0, K / ns)
        s = stim[t]
        _policy(Q, WM, s, w, beta, eps, nA, p_rl, p_wm, p)
        u = uniform_for(seed, agent_id, t)
        a = nA - 1
        cum = 0.0
        for j in range(nA):
            cum += p[j]
            if u < cum:
                a = j
                break
        out_row[t] = a
        r = 1 if a == correct[t] else 0
        _update(Q, WM, s, a, r, alpha, phi, bias_wm, bias_rl, ns, nA)


@njit(parallel=True, cache=True)
def simulate_batch(thetas, agent_ids, seed, new_block, stim, ns_arr, correct,
                   nA, max_ns, out_choices):
    """Simula muchos agentes en paralelo.

    thetas:      (n, 8) parametros de cada agente.
    agent_ids:   (n,) indice global de cada agente (uint64); define su azar.
    out_choices: (n, T) uint8, se llena con la accion elegida (0..nA-1).
    """
    n = thetas.shape[0]
    for i in prange(n):
        _simulate_one(thetas[i], agent_ids[i], seed, new_block, stim, ns_arr,
                      correct, nA, max_ns, out_choices[i])


# ---------------------------------------------------------------------------
# Verosimilitud
# ---------------------------------------------------------------------------
@njit(cache=True)
def loglik_one(theta, choices, new_block, stim, ns_arr, correct, nA, max_ns):
    """NLL de las elecciones de un agente. Elecciones < 0 se omiten (sin respuesta)."""
    alpha = theta[P_ALPHA]; phi = theta[P_PHI]; rho = theta[P_RHO]
    eps = theta[P_EPS]; K = theta[P_K]; beta = theta[P_BETA]
    bias_wm = theta[P_BIAS_WM]; bias_rl = theta[P_BIAS_RL]
    Q = np.empty((max_ns, nA)); WM = np.empty((max_ns, nA))
    p_rl = np.empty(nA); p_wm = np.empty(nA); p = np.empty(nA)
    w = 0.0
    nll = 0.0
    for t in range(stim.shape[0]):
        ns = ns_arr[t]
        if new_block[t]:
            _reset(Q, WM, ns, nA)
            w = rho * min(1.0, K / ns)
        a = choices[t]
        if a < 0:
            continue
        s = stim[t]
        _policy(Q, WM, s, w, beta, eps, nA, p_rl, p_wm, p)
        nll -= np.log(max(p[a], PROB_FLOOR))
        r = 1 if a == correct[t] else 0
        _update(Q, WM, s, a, r, alpha, phi, bias_wm, bias_rl, ns, nA)
    return nll


@njit(parallel=True, cache=True)
def loglik_batch(thetas, choices, new_block, stim, ns_arr, correct, nA, max_ns, out):
    """NLL para muchos pares (parametros, elecciones). choices: (n, T) o (1, T)
    (si es (1, T) se evalua el mismo conjunto de datos con muchos parametros)."""
    n = thetas.shape[0]
    same = choices.shape[0] == 1
    for i in prange(n):
        row = choices[0] if same else choices[i]
        out[i] = loglik_one(thetas[i], row, new_block, stim, ns_arr, correct, nA, max_ns)


# ---------------------------------------------------------------------------
# Repeticion (replay): registro completo de decisiones de un agente
# ---------------------------------------------------------------------------
@njit(cache=True)
def _replay_kernel(theta, choices, new_block, stim, ns_arr, correct, nA, max_ns,
                   o_w, o_prl, o_pwm, o_p, o_q, o_wm, o_r):
    alpha = theta[P_ALPHA]; phi = theta[P_PHI]; rho = theta[P_RHO]
    eps = theta[P_EPS]; K = theta[P_K]; beta = theta[P_BETA]
    bias_wm = theta[P_BIAS_WM]; bias_rl = theta[P_BIAS_RL]
    Q = np.empty((max_ns, nA)); WM = np.empty((max_ns, nA))
    p_rl = np.empty(nA); p_wm = np.empty(nA); p = np.empty(nA)
    w = 0.0
    for t in range(stim.shape[0]):
        ns = ns_arr[t]
        if new_block[t]:
            _reset(Q, WM, ns, nA)
            w = rho * min(1.0, K / ns)
        s = stim[t]
        _policy(Q, WM, s, w, beta, eps, nA, p_rl, p_wm, p)
        o_w[t] = w
        for j in range(nA):
            o_prl[t, j] = p_rl[j]; o_pwm[t, j] = p_wm[j]; o_p[t, j] = p[j]
            o_q[t, j] = Q[s, j]; o_wm[t, j] = WM[s, j]
        a = choices[t]
        if a < 0:
            o_r[t] = -1
            continue
        r = 1 if a == correct[t] else 0
        o_r[t] = r
        _update(Q, WM, s, a, r, alpha, phi, bias_wm, bias_rl, ns, nA)


def replay(theta, choices, plan):
    """Reconstruye el registro completo de un agente.

    Devuelve un dict de arreglos por ensayo: w (peso de WM), p_rl, p_wm, p
    (politicas antes de elegir), q y wm (valores del estimulo presentado antes
    de actualizar), choice, reward y p_choice (probabilidad de la accion elegida).
    """
    T = plan["stim"].shape[0]
    nA = int(plan["n_actions"]); max_ns = int(plan["max_ns"])
    o = {k: np.empty((T, nA)) for k in ("p_rl", "p_wm", "p", "q", "wm")}
    o["w"] = np.empty(T); o["reward"] = np.empty(T, dtype=np.int8)
    ch = np.asarray(choices, dtype=np.int16)
    _replay_kernel(np.asarray(theta, dtype=np.float64), ch, plan["new_block"],
                   plan["stim"], plan["ns"], plan["correct"], nA, max_ns,
                   o["w"], o["p_rl"], o["p_wm"], o["p"], o["q"], o["wm"], o["reward"])
    o["choice"] = ch
    o["p_choice"] = np.where(ch >= 0, o["p"][np.arange(T), np.clip(ch, 0, None)], np.nan)
    return o


# ---------------------------------------------------------------------------
# Envolturas de uso sencillo
# ---------------------------------------------------------------------------
def make_theta(alpha, phi, rho, epsilon, K, beta=17.0, bias_wm=0.0, bias_rl=1.0):
    """Vector de parametros en el orden que usa el nucleo."""
    return np.array([alpha, phi, rho, epsilon, K, beta, bias_wm, bias_rl], dtype=np.float64)


def simulate(thetas, agent_ids, seed, plan, out=None):
    thetas = np.ascontiguousarray(np.atleast_2d(thetas), dtype=np.float64)
    agent_ids = np.ascontiguousarray(agent_ids, dtype=np.uint64)
    T = plan["stim"].shape[0]
    if out is None:
        out = np.empty((thetas.shape[0], T), dtype=np.uint8)
    simulate_batch(thetas, agent_ids, np.uint64(seed), plan["new_block"], plan["stim"],
                   plan["ns"], plan["correct"], int(plan["n_actions"]), int(plan["max_ns"]), out)
    return out


def nll(thetas, choices, plan):
    thetas = np.ascontiguousarray(np.atleast_2d(thetas), dtype=np.float64)
    choices = np.ascontiguousarray(np.atleast_2d(choices), dtype=np.int16)
    out = np.empty(thetas.shape[0])
    loglik_batch(thetas, choices, plan["new_block"], plan["stim"], plan["ns"],
                 plan["correct"], int(plan["n_actions"]), int(plan["max_ns"]), out)
    return out
