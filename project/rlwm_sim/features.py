"""Metricas de desempeno (caracteristicas) por agente, calculadas a partir de
sus elecciones y del plan de la tarea. No usan el modelo ni sus parametros:
solo lo que veria un investigador en datos reales (aciertos y elecciones).

Las metricas se calculan por tamano de conjunto (TC). Si un TC tiene varios
bloques (fases 1 y 2), se acumulan juntos. Definiciones (por TC):

  acc                 proporcion de aciertos
  acc_early           aciertos en las presentaciones 2 y 3 de cada estimulo
  acc_late            aciertos en las ultimas 3 presentaciones
  first_correct_iter  presentacion del primer acierto, promedio por estimulo (P+1 si nunca)
  criterion_iter      presentacion en que se completan 3 aciertos seguidos (P+1 si nunca)
  acc_delay1/2/3p     aciertos segun ensayos desde el ultimo acierto de ese estimulo (1, 2, 3 o mas)
  acc_prev1/2/3p      aciertos segun cuantas veces ya acerto ese estimulo (1, 2, 3 o mas)
  win_stay            P(repetir la respuesta | la presentacion anterior del estimulo fue acierto)
  lose_shift          P(cambiar de respuesta | la presentacion anterior del estimulo fue error)
  persev_err_rate     proporcion de errores que repiten una respuesta ya marcada como incorrecta
  err_after_learned   proporcion de errores despues de alcanzar el criterio en ese estimulo
  max_streak_correct  racha mas larga de aciertos seguidos dentro de un bloque
  max_streak_error    racha mas larga de errores seguidos dentro de un bloque
  prop_learned        proporcion de estimulos que alcanzan el criterio
  last_error_iter     presentacion del ultimo error, promedio por estimulo (0 si nunca fallo)
Globales:
  acc_total           proporcion de aciertos en toda la fase
  setsize_effect      acc del TC menor menos acc del TC mayor

Ensayos sin respuesta (eleccion -1, solo en datos reales) cuentan como error,
no cuentan para win_stay/lose_shift ni como error perseverativo.

Una metrica condicional sin ensayos que cumplan la condicion vale NaN
(por ejemplo acc_delay3p si el agente nunca tuvo ese retraso).
Referencias: Collins y Frank (2012, Fig. 2 y 3), Collins et al. (2014) y la tesis del usuario.
"""
import numpy as np
from numba import njit, prange

PER_TC = ("acc", "acc_early", "acc_late", "first_correct_iter", "criterion_iter",
          "acc_delay1", "acc_delay2", "acc_delay3p", "acc_prev1", "acc_prev2", "acc_prev3p",
          "win_stay", "lose_shift", "persev_err_rate", "err_after_learned",
          "max_streak_correct", "max_streak_error", "prop_learned", "last_error_iter")
GLOBAL = ("acc_total", "setsize_effect")
CRITERION = 3  # aciertos seguidos para considerar aprendido un estimulo
NF = len(PER_TC)
NSLOT = 33


def tc_values(plan):
    return np.unique(plan["ns"]).astype(np.int64)


def feature_names(plan):
    names = [f"{f}_tc{int(n)}" for n in tc_values(plan) for f in PER_TC]
    return names + list(GLOBAL)


def plan_aux(plan):
    """Arreglos auxiliares por ensayo: indice de TC y presentaciones del bloque."""
    tcs = tc_values(plan)
    tc_idx = np.searchsorted(tcs, plan["ns"]).astype(np.int64)
    block = plan["block"].astype(np.int64)
    pres = np.zeros(len(block), dtype=np.int64)
    for b in np.unique(block):
        m = block == b
        pres[m] = plan["iteration"][m].max()
    return tc_idx, pres, len(tcs)


@njit(cache=True, inline="always")
def _rate(num, den):
    return num / den if den > 0 else np.nan


@njit(cache=True)
def _features_one(ch, new_block, stim, ns_arr, correct, iteration, tc_idx, pres,
                  n_tc, max_ns, out):
    acc = np.zeros((n_tc, NSLOT))
    last_choice = np.empty(max_ns, np.int64); last_reward = np.empty(max_ns, np.int64)
    n_corr = np.empty(max_ns, np.int64); last_corr_pos = np.empty(max_ns, np.int64)
    wrong_mask = np.empty(max_ns, np.int64); consec = np.empty(max_ns, np.int64)
    first_it = np.empty(max_ns, np.int64); crit_it = np.empty(max_ns, np.int64)
    last_err_it = np.empty(max_ns, np.int64)
    T = stim.shape[0]
    block_start = 0
    cur_c = 0; cur_e = 0
    for t in range(T):
        if new_block[t]:
            block_start = t
            cur_c = 0; cur_e = 0
            for i in range(max_ns):
                last_choice[i] = -1; last_reward[i] = -1; n_corr[i] = 0
                last_corr_pos[i] = -1; wrong_mask[i] = 0; consec[i] = 0
                first_it[i] = 0; crit_it[i] = 0; last_err_it[i] = 0
        k = tc_idx[t]; P = pres[t]; s = stim[t]; it = iteration[t]
        pos = t - block_start
        a = np.int64(ch[t])  # -1 = sin respuesta (solo en datos reales)
        r = 1 if (a >= 0 and a == correct[t]) else 0
        A = acc[k]
        A[0] += r; A[1] += 1
        if it >= 2 and it <= 3:
            A[2] += r; A[3] += 1
        if it >= P - 2:
            A[4] += r; A[5] += 1
        if n_corr[s] > 0:
            d = pos - last_corr_pos[s]
            j = 9 if d == 1 else (11 if d == 2 else 13)
            A[j] += r; A[j + 1] += 1
        if n_corr[s] >= 1:
            j = 15 if n_corr[s] == 1 else (17 if n_corr[s] == 2 else 19)
            A[j] += r; A[j + 1] += 1
        if last_choice[s] >= 0 and a >= 0:
            if last_reward[s] == 1:
                A[22] += 1
                if a == last_choice[s]:
                    A[21] += 1
            else:
                A[24] += 1
                if a != last_choice[s]:
                    A[23] += 1
        if crit_it[s] > 0:
            A[28] += 1
            if r == 0:
                A[27] += 1
        if r == 0:
            A[26] += 1
            if a >= 0:
                if (wrong_mask[s] >> a) & 1:
                    A[25] += 1
                wrong_mask[s] |= (1 << a)
            last_err_it[s] = it
            consec[s] = 0
            cur_e += 1; cur_c = 0
        else:
            n_corr[s] += 1
            last_corr_pos[s] = pos
            if first_it[s] == 0:
                first_it[s] = it
            consec[s] += 1
            if consec[s] == CRITERION and crit_it[s] == 0:
                crit_it[s] = it
            cur_c += 1; cur_e = 0
        if cur_c > A[29]:
            A[29] = cur_c
        if cur_e > A[30]:
            A[30] = cur_e
        last_choice[s] = a; last_reward[s] = r
        # cierre del bloque: resumen por estimulo
        if t == T - 1 or new_block[t + 1]:
            nsb = ns_arr[t]
            for i in range(nsb):
                A[7] += 1
                A[6] += first_it[i] if first_it[i] > 0 else P + 1
                A[8] += crit_it[i] if crit_it[i] > 0 else P + 1
                if crit_it[i] > 0:
                    A[31] += 1
                A[32] += last_err_it[i]
    tot_num = 0.0; tot_den = 0.0
    for k in range(n_tc):
        A = acc[k]; o = k * NF
        out[o + 0] = _rate(A[0], A[1])
        out[o + 1] = _rate(A[2], A[3])
        out[o + 2] = _rate(A[4], A[5])
        out[o + 3] = _rate(A[6], A[7])
        out[o + 4] = _rate(A[8], A[7])
        out[o + 5] = _rate(A[9], A[10])
        out[o + 6] = _rate(A[11], A[12])
        out[o + 7] = _rate(A[13], A[14])
        out[o + 8] = _rate(A[15], A[16])
        out[o + 9] = _rate(A[17], A[18])
        out[o + 10] = _rate(A[19], A[20])
        out[o + 11] = _rate(A[21], A[22])
        out[o + 12] = _rate(A[23], A[24])
        out[o + 13] = _rate(A[25], A[26])
        out[o + 14] = _rate(A[27], A[28])
        out[o + 15] = A[29]
        out[o + 16] = A[30]
        out[o + 17] = _rate(A[31], A[7])
        out[o + 18] = _rate(A[32], A[7])
        tot_num += A[0]; tot_den += A[1]
    g = n_tc * NF
    out[g] = _rate(tot_num, tot_den)
    out[g + 1] = _rate(acc[0, 0], acc[0, 1]) - _rate(acc[n_tc - 1, 0], acc[n_tc - 1, 1])


@njit(parallel=True, cache=True)
def features_batch(choices, new_block, stim, ns_arr, correct, iteration, tc_idx, pres,
                   n_tc, max_ns, out):
    for i in prange(choices.shape[0]):
        _features_one(choices[i], new_block, stim, ns_arr, correct, iteration,
                      tc_idx, pres, n_tc, max_ns, out[i])


def compute_features(choices, plan):
    """(agentes, 90) elecciones -> (agentes, n_caracteristicas) float32."""
    choices = np.ascontiguousarray(np.atleast_2d(choices))
    tc_idx, pres, n_tc = plan_aux(plan)
    out = np.empty((choices.shape[0], n_tc * NF + len(GLOBAL)), dtype=np.float32)
    features_batch(choices, plan["new_block"], plan["stim"].astype(np.int64),
                   plan["ns"].astype(np.int64), plan["correct"].astype(np.int64),
                   plan["iteration"].astype(np.int64), tc_idx, pres, n_tc,
                   int(plan["max_ns"]), out)
    return out
