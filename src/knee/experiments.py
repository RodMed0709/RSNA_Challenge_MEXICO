import json
import os
import random
import re
import shutil
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT, USERS_ROOT, current_user

CLAIMS_ROOT = USERS_ROOT / ".claims"
EXP_PATTERN = re.compile(r"^exp(\d{3})_[a-z0-9_]+$")


def seed_everything(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_info(repo=REPO_ROOT):
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""

    return {
        "commit": run("rev-parse", "--short", "HEAD") or None,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD") or None,
        "src_dirty": bool(run("status", "--porcelain", "--", "src", "splits")),
    }


def _numbers_in(folder):
    if not folder.exists():
        return set()
    return {int(m.group(1)) for p in folder.iterdir() if (m := EXP_PATTERN.match(p.name))}


def used_numbers():
    numbers = _numbers_in(CLAIMS_ROOT) | _numbers_in(REPO_ROOT / "experiments")
    if USERS_ROOT.exists():
        for user in USERS_ROOT.iterdir():
            numbers |= _numbers_in(user / "experiments")
    registry = REPO_ROOT / "docs" / "EXPERIMENTOS.md"
    if registry.exists():
        numbers |= {int(n) for n in re.findall(r"^\|\s*exp(\d{3})", registry.read_text(encoding="utf-8"), re.M)}
    return numbers


def find_experiment(exp_id, root=None):
    root = Path(root) if root else REPO_ROOT / "experiments"
    matches = sorted(root.glob(f"{exp_id}_*")) if re.fullmatch(r"exp\d{3}", exp_id) else [root / exp_id]
    matches = [m for m in matches if m.is_dir()]
    if not matches:
        raise FileNotFoundError(f"No encontré {exp_id} en {root}")
    return matches[0]


def nuevo_experimento(tag, padre, idea=""):
    tag = re.sub(r"[^a-z0-9_]+", "_", tag.lower()).strip("_")
    if not tag:
        raise ValueError("El tag no puede estar vacío, por ejemplo: 'dinov2_base'")
    parent_dir = find_experiment(padre)
    CLAIMS_ROOT.mkdir(parents=True, exist_ok=True)
    number = max(used_numbers(), default=0) + 1
    while True:
        name = f"exp{number:03d}_{tag}"
        try:
            (CLAIMS_ROOT / name).mkdir()
            break
        except FileExistsError:
            number += 1
    (CLAIMS_ROOT / name / "owner.txt").write_text(f"{current_user()} {datetime.now(timezone.utc).isoformat()}\n")

    exp_dir = REPO_ROOT / "experiments" / name
    exp_dir.mkdir(parents=True)
    notebooks = sorted(parent_dir.glob("exp*.ipynb"))
    if notebooks:
        _copy_notebook(notebooks[0], exp_dir / f"{name}.ipynb", name, parent_dir.name[:6])
    template = (REPO_ROOT / "experiments" / "_plantilla" / "SPEC.md").read_text(encoding="utf-8")
    spec = (template.replace("expNNN_tag", name).replace("<autor>", current_user())
            .replace("<padre>", parent_dir.name).replace("<fecha>", date.today().isoformat())
            .replace("<idea>", idea or "<describe la idea en una línea>"))
    (exp_dir / "SPEC.md").write_text(spec, encoding="utf-8")
    (REPO_ROOT / "results" / name).mkdir(parents=True, exist_ok=True)
    _registry_add(name, parent_dir.name[:6], idea)
    print(f"Listo: {name}")
    print(f"1. Llena la SPEC: experiments/{name}/SPEC.md (secciones 1 y 2)")
    print(f"2. Cambia solo las celdas del PLAN en experiments/{name}/{name}.ipynb")
    print(f"3. Rama sugerida: git switch -c {name}")
    return exp_dir


def _copy_notebook(source, target, name, parent):
    notebook = json.loads(Path(source).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"], cell["execution_count"] = [], None
        if "parameters" in cell.get("metadata", {}).get("tags", []):
            text = "".join(cell["source"])
            text = re.sub(r'^EXP = ".*"$', f'EXP = "{name}"', text, flags=re.M)
            text = re.sub(r'^PADRE = ".*"$', f'PADRE = "{parent}"', text, flags=re.M)
            text = re.sub(r"^SMOKE = .*$", "SMOKE = True", text, flags=re.M)
            cell["source"] = text.splitlines(keepends=True)
    Path(target).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _registry_add(name, parent, idea):
    registry = REPO_ROOT / "docs" / "EXPERIMENTOS.md"
    row = f"| {name[:6]} | {name[7:]} | {current_user()} | {parent} | {idea or '—'} | spec | | | | |\n"
    text = registry.read_text(encoding="utf-8") if registry.exists() else ""
    registry.write_text(text.rstrip("\n") + "\n" + row, encoding="utf-8")


def guardar_metricas(results_dir, metrics):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "metrics.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)
    return path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def leer_metricas(exp_id, root=None):
    results = Path(root) if root else REPO_ROOT / "results"
    path = find_experiment(exp_id, results) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def tabla_experimentos(users_root=None):
    rows = []
    for path in sorted(Path(users_root or USERS_ROOT).glob("*/results/exp*/metrics.json")):
        try:
            m = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        folds = m.get("per_fold", {})
        rows.append({
            "exp": path.parent.name,
            "autor": m.get("author") or path.parents[2].name,
            "padre": m.get("parent"),
            "fase": m.get("phase"),
            "folds": ",".join(sorted(folds)),
            "pseudo_f0": folds.get("0", {}).get("pseudo_auc"),
            "oof_pseudo": m.get("oof_pseudo_auc"),
            "gold": m.get("gold_auc"),
            "decision": m.get("decision_auto"),
            "pesos": [f.get("ckpt_path") for f in folds.values() if f.get("ckpt_path")],
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values(["oof_pseudo", "pseudo_f0"], ascending=False, na_position="last").reset_index(drop=True)


def pesos(exp_id, users_root=None):
    table = tabla_experimentos(users_root)
    row = table[table.exp.str.startswith(exp_id)]
    if row.empty:
        raise FileNotFoundError(f"No hay metrics.json para {exp_id}")
    return list(row.iloc[0].pesos)


def decision(delta_pseudo, delta_gold):
    if delta_gold is not None and np.isfinite(delta_gold) and delta_gold < -0.02:
        return "killed"
    if delta_pseudo is None or not np.isfinite(delta_pseudo):
        return "sin_padre"
    if delta_pseudo >= 0.005:
        return "promising"
    if delta_pseudo > -0.005:
        return "neutral"
    return "didnt"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Experimentos del equipo")
    sub = parser.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("nuevo")
    new.add_argument("tag")
    new.add_argument("--padre", required=True)
    new.add_argument("--idea", default="")
    sub.add_parser("tabla")
    args = parser.parse_args()
    if args.cmd == "nuevo":
        nuevo_experimento(args.tag, args.padre, args.idea)
    else:
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(tabla_experimentos().drop(columns="pesos"))
