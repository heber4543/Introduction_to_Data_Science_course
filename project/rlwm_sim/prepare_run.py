"""Paso 1. Prepara una corrida: crea carpetas, genera el plan de la tarea y
escribe el manifiesto. Se ejecuta UNA vez antes de lanzar los fragmentos.

Uso:  python -m rlwm_sim.prepare_run configs/fase3_rejilla001_tesis.json
"""
import json
import sys
import time

from .environment import make_plan, save_plan, load_plan, plan_hash
from .grid import Grid
from .runio import load_config, run_paths, code_hash, versions


def main(cfg_path):
    cfg = load_config(cfg_path)
    paths = run_paths(cfg)
    for k in ("run_dir", "shards", "logs"):
        paths[k].mkdir(parents=True, exist_ok=True)

    if paths["manifest"].exists():
        print(f"Ya existe {paths['manifest']}. No se sobrescribe una corrida existente.")
        print("Si quieres otra corrida, cambia run_dir en la configuracion.")
        return 1

    if paths["plan_npz"].exists():
        plan = load_plan(paths["plan_npz"])
        print("Se reutiliza el plan existente.")
    else:
        plan = make_plan(cfg["design"], cfg["plan_seed"])
        save_plan(plan, paths["plan_npz"], paths["plan_json"])
        print("Plan generado:")
        for b in plan["meta"]["blocks"]:
            print(f"  bloque {b['block']}: TC={b['set_size']}, presentaciones={b['presentations']},"
                  f" respuestas={b['mapping']}, distancia a retraso uniforme={b['delay_tv_from_uniform']}")

    if cfg.get("sequence", {}).get("mode") == "random_per_agent":
        print("Nota: esta corrida usa una secuencia aleatoria por agente. El plan anterior es solo")
        print("una plantilla (tamanos de conjunto y presentaciones); cada agente genera la suya.")
    grid = Grid(cfg["grid"])
    R = int(cfg.get("replicates", 1))
    manifest = {
        "run_name": cfg["run_name"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "code_hash": code_hash(),
        "versions": versions(),
        "plan_hash": plan_hash(plan),
        "n_trials": int(plan["stim"].shape[0]),
        "grid_axes": grid.names,
        "grid_shape": list(grid.shape),
        "n_grid_points": grid.n_total,
        "replicates": R,
        "n_agents": grid.n_total * R,
        "n_shards": int(cfg["n_shards"]),
        "sim_seed": int(cfg["sim_seed"]),
        "model_fixed": grid.fixed,
    }
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in cfg.items()}, f, indent=2, ensure_ascii=False)
    with open(paths["manifest"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Puntos de la rejilla: {grid.n_total:,}  replicas: {R}  agentes totales: {grid.n_total * R:,}"
          f"  ensayos por agente: {manifest['n_trials']}")
    gb = grid.n_total * R * manifest["n_trials"] / 1e9
    print(f"Espacio aproximado de las elecciones: {gb:.1f} GB")
    print(f"Fragmentos: {cfg['n_shards']} (indices de arreglo SLURM 0-{cfg['n_shards'] - 1})")
    print(f"Corrida lista en {paths['run_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
